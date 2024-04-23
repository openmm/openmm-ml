# NequIP models in OpenMM-ML

This directory contains examples for running simulations using a NequIP potential.

## Installation

First install the `openmm-torch` and `nnpops` packages from conda-forge:

```
conda install -c conda-forge openmm-torch nnpops
```

Then install NequIP development branch and this version of `openmm-ml` using pip:

```
pip install git+https://github.com/mir-group/nequip@develop
pip install git+https://github.com/sef43/openmm-ml@nequip
```

## Usage

Once you have a deployed trained NequIP model you can use it as the potential in OpenMM-ML:

```python
from openmmml import MLPotential

# Create a System with NequIP MLP

# Need to specify the unit conversion factors from the NequIP model units to OpenMM units.
# e.g.:
# Distance: model is in Angstrom, OpenMM is in nanometers
A_to_nm = 0.1
# Energy: model is in kcal/mol, OpenMM is in kJ/mol
kcal_to_kJ_per_mol = 4.184

potential = MLPotential('nequip', 
                        modelPath='example_model_deployed.pth',
                        lengthScale=A_to_nm,
                        energyScale=kcal_to_kJ_per_mol)

system = potential.createSystem(topology)
```

## Example

### run_nequip.ipynb
Runs a simulation using the model created by NequIP example [config/example.yaml](https://github.com/mir-group/nequip/blob/main/configs/example.yaml). It is available as a Python script: [`run_nequip.py`](run_nequip.py) and a Jupyter notebook [`run_nequip.ipynb`](run_nequip.ipynb) which can be run on Colab.
