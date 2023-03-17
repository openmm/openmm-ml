# MACE models in OpenMM

This directory contains examples for running simulations using a [MACE](https://github.com/ACEsuit/mace) potential.

## Install

First you will need `openmm` and `openmm-torch` from conda-forge
```
mamba install -c conda-forge openmm openmm-torch
```

Then you will need to install MACE and this fork of openmm-ml with pip:

```
pip install git+https://github.com/ACEsuit/mace
pip install git+https://github.com/sef43/openmm-ml@mace
```

## Usage
```python
from openmmml import MLPotential

potential = MLPotential('mace', model_path='path/to/model')

system = potential.createSystem(topology)
```


## Examples
There are three examples:
- [run_mace.py](run_mace.py) runs a small molecule with pure MACE ML potential.
- [run_mace_mixed.py](run_mace_mixed.py) runs a small molecule in a water box where the small molecule's intra-molecular forces use MACE and the rest of the system in MM.
- [run_water_box.py](run_water_box.py) runs a water box with pure MACE ML potential.

## References
The example pre-trained model is from here: https://github.com/jharrymoore/mace-openmm/tree/main/tests/example_data
The MACE repo is here: https://github.com/ACEsuit/mace 
