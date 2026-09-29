"""Verified atomic copies and resumable ZIPs, using only the Python standard library.

SQLite checkpoints and partial ZIPs are local working files, never deposit files.
A saved ZIP central directory allows recovery after SIGKILL while appending:
restore the last committed directory, truncate the unfinished batch, and append
only the remaining members. Completed batches are never recompressed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


BLOCK_SIZE = 1024 * 1024


def file_signature(path: Path) -> str:
    stat = path.stat()
    return json.dumps([stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class Member:
    name: str
    path: Path
    digest: str
    signature: str


class ResumeStore:
    """One process at a time; source changes and edited scientific copies fail closed."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir.resolve()

    def __enter__(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock = (self.state_dir / "lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("Another archive process is using this resume directory.")
        self.db = sqlite3.connect(self.state_dir / "progress.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS copies (
                target TEXT PRIMARY KEY, source TEXT NOT NULL,
                source_signature TEXT NOT NULL, target_signature TEXT NOT NULL,
                digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS archives (
                target TEXT PRIMARY KEY, signature TEXT NOT NULL,
                directory_offset INTEGER NOT NULL, directory BLOB NOT NULL,
                complete INTEGER NOT NULL DEFAULT 0,
                final_signature TEXT, digest TEXT
            );
        """)
        self.db.commit()
        self.copied = self.skipped = 0
        return self

    def __exit__(self, *_):
        self.db.close()
        self.lock.close()

    def _remember(self, target, source, source_signature, digest):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO copies VALUES (?, ?, ?, ?, ?)",
                (str(target), str(source), source_signature, file_signature(target), digest),
            )

    def ensure_file(self, source: Path, target: Path, *, verify=False, require_existing=False) -> str:
        source, target = source.resolve(), target.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        source_signature = file_signature(source)
        row = self.db.execute("SELECT * FROM copies WHERE target=?", (str(target),)).fetchone()
        if row and row["source"] != str(source):
            raise RuntimeError(f"Resume source changed for {target}; use a fresh output directory.")
        if (row and not verify and target.is_file()
                and row["source_signature"] == source_signature
                and row["target_signature"] == file_signature(target)):
            self.skipped += 1
            return row["digest"]

        if target.exists():
            digest = sha256_file(source)
            if row and row["digest"] != digest:
                raise RuntimeError(f"Scientific source changed: {source}; use a fresh output directory.")
            if sha256_file(target) != digest:
                raise RuntimeError(f"Refusing to overwrite a different scientific file: {target}")
            if file_signature(source) != source_signature:
                raise RuntimeError(f"Source changed during verification: {source}")
            self._remember(target, source, source_signature, digest)
            self.skipped += 1
            return digest
        if require_existing:
            raise FileNotFoundError(f"Archive file is missing: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".archive-part")
        digest = hashlib.sha256()
        with source.open("rb") as original, temporary.open("wb") as copied:
            for block in iter(lambda: original.read(BLOCK_SIZE), b""):
                digest.update(block)
                copied.write(block)
            copied.flush()
            os.fsync(copied.fileno())
        if file_signature(source) != source_signature:
            raise RuntimeError(f"Source changed during copying: {source}")
        expected = digest.hexdigest()
        if row and row["digest"] != expected:
            raise RuntimeError(f"Scientific source changed: {source}; use a fresh output directory.")
        if sha256_file(temporary) != expected:
            raise RuntimeError(f"Copied bytes differ from source: {source}")
        shutil.copystat(source, temporary)
        os.replace(temporary, target)
        sync_directory(target.parent)
        self._remember(target, source, source_signature, expected)
        self.copied += 1
        return expected

    def ensure_bytes(self, content: bytes, target: Path, *, require_existing=False) -> str:
        """Update owned generated documentation; protect unrelated or edited files."""
        target = target.resolve()
        digest = hashlib.sha256(content).hexdigest()
        if target.exists():
            current = sha256_file(target)
            if current == digest:
                self._remember(target, "generated", digest, digest)
                return digest
            row = self.db.execute("SELECT * FROM copies WHERE target=?", (str(target),)).fetchone()
            if require_existing or not row or row["source"] != "generated" or row["digest"] != current:
                raise RuntimeError(f"Refusing to overwrite changed documentation: {target}")
        elif require_existing:
            raise FileNotFoundError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".archive-part")
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        sync_directory(target.parent)
        self._remember(target, "generated", digest, digest)
        return digest

    def _checkpoint(self, target: Path, partial: Path, signature: str):
        # start_dir is the offset of the central directory in Python's ZipFile.
        with zipfile.ZipFile(partial) as archive:
            offset = archive.start_dir
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
            stream.seek(offset)
            directory = stream.read()
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO archives "
                "(target, signature, directory_offset, directory, complete) VALUES (?, ?, ?, ?, 0)",
                (str(target), signature, offset, directory),
            )

    def _verify_zip(self, path: Path, members: list[Member]):
        expected = {member.name: member for member in members}
        started = time.monotonic()
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise RuntimeError(f"ZIP membership differs from the selection: {path}")
            for name in names:
                member = expected[name]
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
                        digest.update(block)
                if digest.hexdigest() != member.digest:
                    raise RuntimeError(f"ZIP bytes differ from scientific source: {name}")
                if time.monotonic() - started > 15:
                    print(f"Verifying {path.name}: {name}", flush=True)
                    started = time.monotonic()

    def _write_member(self, archive: zipfile.ZipFile, member: Member):
        # ZipFile.write uses 8 KiB copy blocks. Larger blocks avoid excessive
        # filesystem calls for the large ORCA text outputs and Hessians.
        info = zipfile.ZipInfo.from_file(member.path, member.name)
        info.compress_type = archive.compression
        info._compresslevel = archive.compresslevel  # Python 3.10+ compatibility
        with member.path.open("rb") as original, archive.open(info, "w") as compressed:
            shutil.copyfileobj(original, compressed, BLOCK_SIZE)

    def ensure_zip(self, target: Path, members: list[Member], *, level=6,
                   verify=False, require_existing=False,
                   batch_files=100, batch_bytes=256 * 1024 * 1024):
        target = target.resolve()
        members = sorted(members, key=lambda member: member.name)
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member names")
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"Unsafe archive member: {name}")
        signature = hashlib.sha256(
            json.dumps([level, [(m.name, m.digest) for m in members]], separators=(",", ":")).encode()
        ).hexdigest()
        row = self.db.execute("SELECT * FROM archives WHERE target=?", (str(target),)).fetchone()
        if row and row["signature"] != signature:
            raise RuntimeError(f"ZIP selection changed: {target}; use a fresh output directory.")
        if row and row["complete"] and target.exists():
            if row["final_signature"] != file_signature(target):
                if sha256_file(target) != row["digest"]:
                    raise RuntimeError(f"Completed ZIP was modified: {target}")
            if verify:
                self._verify_zip(target, members)
            print(f"ZIP verified, unchanged: {target.name}", flush=True)
            return
        if require_existing:
            raise RuntimeError(f"ZIP is missing or not finished: {target}")

        partial = self.state_dir / (hashlib.sha256(str(target).encode()).hexdigest()[:16] + ".zip.part")
        if row and not partial.exists() and target.exists():
            # Interrupted after the final rename, before the completion DB commit.
            self._verify_zip(target, members)
            self._mark_complete(target)
            return
        if target.exists():
            raise RuntimeError(f"Refusing to replace an unowned/incomplete ZIP: {target}")
        if row:
            if not partial.exists():
                raise RuntimeError(f"ZIP checkpoint exists but its partial file is missing: {partial}")
            with partial.open("r+b") as stream:
                stream.seek(row["directory_offset"])
                stream.write(row["directory"])
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
        else:
            with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=level, allowZip64=True):
                pass
            self._checkpoint(target, partial, signature)

        with zipfile.ZipFile(partial) as archive:
            completed = archive.namelist()
        if completed != names[:len(completed)]:
            raise RuntimeError(f"Checkpoint members differ from the planned prefix: {target}")
        position = len(completed)
        print(f"ZIP {target.name}: resuming at {position}/{len(members)} files", flush=True)
        while position < len(members):
            next_position, bytes_in_batch = position, 0
            with zipfile.ZipFile(partial, "a", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=level, allowZip64=True) as archive:
                while next_position < len(members):
                    member = members[next_position]
                    if file_signature(member.path) != member.signature:
                        raise RuntimeError(f"Archive input changed: {member.path}")
                    self._write_member(archive, member)
                    if file_signature(member.path) != member.signature:
                        raise RuntimeError(f"Archive input changed during compression: {member.path}")
                    bytes_in_batch += member.path.stat().st_size
                    next_position += 1
                    if next_position - position >= batch_files or bytes_in_batch >= batch_bytes:
                        break
            self._checkpoint(target, partial, signature)
            position = next_position
            print(f"ZIP {target.name}: checkpoint {position}/{len(members)} files", flush=True)

        self._verify_zip(partial, members)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, target)
        sync_directory(target.parent)
        self._mark_complete(target)
        print(f"ZIP finished: {target.name} ({target.stat().st_size / 1e9:.3f} GB)", flush=True)

    def _mark_complete(self, target: Path):
        with self.db:
            self.db.execute(
                "UPDATE archives SET complete=1, final_signature=?, digest=? WHERE target=?",
                (file_signature(target), sha256_file(target), str(target)),
            )
