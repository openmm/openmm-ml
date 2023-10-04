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


## Example
There is an example which runs a simulation of an alanine dipeptide molecule using a MACE model trained on the ANI-1x dataset from https://doi.org/10.1063/5.0155322. It is avaible as a python script [`run_mace.py`](run_mace.py) and a Jupyter notebook [`run_mace.ipynb`](run_mace.ipynb) which can be run on Colab.


## References
The example pre-trained model is from https://doi.org/10.1063/5.0155322 avaiable to download at https://github.com/ACEsuit/mace/blob/docs/docs/examples/ANI_trained_MACE.zip.

The MACE repo is here: https://github.com/ACEsuit/mace 
