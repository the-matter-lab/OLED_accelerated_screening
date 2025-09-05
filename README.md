# Accelerated multifidelity pipeline for OLED screening

<p>
This repository contains a customized Python-based pipeline for high-throughput quantum chemical calculations at different levels of theory.
It is designed and tested on the <strong>Niagara</strong> HPC cluster and keeps clean, per-molecule
<strong>metadata</strong> plus centralized <code>logs/</code> for easy auditing and recovery.
It’s also <strong>El Agente–friendly</strong>.
</p> 
 


---

## Step-by-Step Workflow


### 1. Generate XYZ coordinates from smiles and filter
```bash
python obabel.py
python filter_obabel.py
```


> **ORCA Dependency**  
> This project uses the [ORCA quantum chemistry software](https://orcaforum.kofo.mpg.de), which is **not included** in this repository.  
> To run ORCA-dependent scripts, please download ORCA separately from the official website after agreeing to its academic license terms.  
> We do not distribute ORCA binaries, source, or installation tools. 
> **Installation time**  
> The main quantum chemistry packages (ORCA, CREST, TheoDORE, etc.) were accessed via pre-installed HPC modules, so installation time on a desktop computer is not applicable.


