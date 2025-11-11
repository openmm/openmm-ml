# User Guide

## Introduction

OpenMM-ML is a high level API for using machine learning interatomic potentials (MLIPs) in OpenMM simulations.  With
just a few lines of code, you can set up a simulation that uses a standard, pretrained model to represent some or all of
the interactions in a system.

OpenMM-ML does not provide the models itself.  It relies on other packages such as [MACE](https://github.com/ACEsuit/mace),
[NequIP](https://github.com/mir-group/nequip), [TorchANI](https://github.com/aiqm/torchani), and others for that.  It
provides a common interface for working with them, and handles all the details of connecting them to OpenMM.

## Installation

The stable version of OpenMM-ML can be installed with conda or mamba.

```bash
mamba install -c conda-forge openmm-ml
```

We recommend using mamba, since it is faster and less buggy than conda.

This installs only OpenMM-ML itself, not the packages that provide specific models.  Those must be installed separately
by following the instructions from the package developers.

## Usage

The central class in OpenMM-ML is `MLPotential`.  It represents an MLIP that can be used for simulations.  In the
simplest cases you just provide it the name of the pretrained potential function to use.  You can then call
`createSystem()` to create a `System` object for a simulation.  For example,

```python
from openmmml import MLPotential
potential = MLPotential('ani2x')
system = potential.createSystem(topology)
```

To use a model you provide yourself rather than a standard pretrained one, include the `modelPath` option to point to
the file containing your model.  For example,

```python
potential = MLPotential('mace', modelPath='MACE.model')
```

Other options that are specific to particular types of models are described below.

Rather than simulating the entire model with an MLIP, you can use `createMixedSystem()` to create a `System` where part
is  modeled with the MLIP and the rest is modeled with a conventional force field.  To do this, first create a `System`
that is entirely modeled with the force field.  Then call `createMixedSystem()`, providing it the list of atoms to
replace with the MLIP.

As an example, suppose the `Topology` contains three chains.  Chain 0 is a protein, chain 1 is a ligand, and chain 2 is
solvent.  The following code creates a `System` in which the internal energy of the ligand is computed with ANI2x, while
everything else (including interactions between the ligand and the rest of the system) is computed with Amber14.

```python
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
mm_system = forcefield.createSystem(topology)
chains = list(topology.chains())
ml_atoms = [atom.index for atom in chains[1].atoms()]
potential = MLPotential('ani2x')
ml_system = potential.createMixedSystem(topology, mm_system, ml_atoms)
```

See the API documentation for additional information about the arguments to `createSystem()`  and `createMixedSystem()`.

## Supported Models

OpenMM-ML supports models created with a number of packages.  For each one we list the supported model names (the first
argument to the `MLPotential` constructor), as well as additional arguments that can be passed to the constructor and to
`createSystem()`.

### MACE

The [MACE](https://github.com/ACEsuit/mace) package can be used to create models based on the [MACE architecture](https://arxiv.org/abs/2206.07697).
This includes both pretrained models and custom models you create yourself.  The following model names
are supported.

| Name | Model |
| --- | --- |
| `mace-off23-small`<br>`mace-off23-medium`<br>`mace-off23-large` | Pretrained [MACE-OFF23](https://arxiv.org/abs/2312.15211) models |
| `mace` | Custom MACE models specified with the `modelPath` argument |

When creating MACE models, the following keyword arguments to the `MLPotential` constructor are supported.

| Argument | Description |
| --- | --- |
| `modelPath` | For custom models, the path to the file containing the model |

When using MACE models, the following extra keyword arguments to `createSystem()` and `createMixedSystem()` are supported.

| Argument | Description |
| --- | --- |
| `precision` | The numerical precision of the model. Supported options are `'single'` and `'double'`.  If `None`, the default precision of the model is used. |
 | `returnEnergyType` | Whether to return the interaction energy or the energy including the self-energy.  The default is `'interaction_energy'`. Supported options are `'interaction_energy'` and `'energy'`. |

### AIMNet2

The [aimnet](https://github.com/isayevlab/aimnetcentral) package can be used to create models using the pretrained
[AIMNet2](https://doi.org/10.1039/D4SC08572H) potential.  The following model names
are supported.

| Name | Model |
| --- | --- |
| `aimnet2` | Pretrained AIMNet2 models |

When using AIMNet2 models, the following extra keyword arguments to `createSystem()` and `createMixedSystem()` are supported.

| Argument       | Description |
| --- | --- |
| `charge`       | The total charge of the system.  If omitted, it is assumed to be 0. |
| `multiplicity` | The spin multiplicity of the system.  If omitted, it is assumed to be 1. |

### NequIP

The [NequIP](https://github.com/mir-group/nequip) package can be used to create models based on the [NequIP architecture](https://www.nature.com/articles/s41467-022-29939-5).
That includes models created with [Allegro](https://github.com/mir-group/allegro), which is an extension
package for NequIP.  The following model names are supported.

| Name | Model |
| --- | --- |
| `nequip` | Custom NequIP or Allegro models specified with the `modelPath` argument |

When creating NequIP models, the following keyword arguments to the `MLPotential` constructor are supported.

| Argument | Description |
| --- | --- |
| `modelPath` | The path to the file containing the model |
| `lengthScale` | The conversion factor from the model's length units to nanometers (e.g. 0.1 if the model uses Angstroms) |
| `energyScale` | The conversion factor from the model's energy units to kJ/mol (e.g. 4.184 if the model uses kcal/mol) |

When using NequIP models, the following extra keyword arguments to `createSystem()` and `createMixedSystem()` are supported.

| Argument | Description |
| --- | --- |
| `precision` | The numerical precision of the model. Supported options are `'single'` and `'double'`.  If `None`, the default precision of the model is used. |
 | `atomTypes` | A list of integers corresponding to the NequiIP atom type for each ML atom in the system.  This is only required if the model was trained with custom atom types. If `None`, the atomic number is used to determine the atom type. This list should have the same length as the number of ML atoms in the system. |

### TorchANI

The [TorchANI](https://github.com/aiqm/torchani) package can be used to create models using the pretrained
[ANI-1ccx](https://www.nature.com/articles/s41467-019-10827-4) and [ANI-2x](https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00121)
potentials.  The  [NNPOps](https://github.com/openmm/NNPOps) package can optionally be used to accelerate model
calculations.

Both ANI-1ccx and ANI-2x are ensembles of eight models.  Averaging over all eight  models leads to slightly more accurate
results than any one individually.  You can optionally use only a single model, which leads to a large improvement in
speed at the cost of a small decrease in accuracy.

The following model names are supported.

| Name | Model |
| --- | --- |
| `ani1ccx` | Pretrained ANI-1ccx model |
| `ani2x` | Pretrained ANI-2x model |

When using TorchANI models, the following extra keyword arguments to `createSystem()` and `createMixedSystem()` are supported.

| Argument | Description |
| --- | --- |
| `implementation` | Selects the implementation to use.  Supported options are `'nnpops'` (tends to be faster for small systems) and `'torchani'` (tends to be faster for large systems). |
 | `modelIndex` | The index of the model within the ensemble to use.  If it is `None`, the full ensemble of eight models is used. |

### DeePMD-kit

The [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) package can be used to create models based on the
[Deep Potential](https://arxiv.org/abs/1707.01478) architecture.  The following model names are supported.

| Name | Model |
| --- | --- |
| `deepmd` | Custom models specified with the `model` argument |

When creating DeePMD-kit models, the following keyword arguments to the `MLPotential` constructor are supported.

| Argument | Description |
| --- | --- |
| `model` | The path to the file containing the model |
| `coordinatesCoefficient` | The conversion factor between the model's length units and nanometers.  The default value is 10, corresponding to DeePMD-kit's default units of Angstroms. |
| `forceCoefficient` | The conversion factor between the model's force units and kJ/mol/nm.  The default value is 964.8792534459, corresponding to DeePMD-kit's default units of eV/Å. |
| `energyCoefficient` | The conversion factor between the model's energy units and kJ/mol.  The default value is 96.48792534459, corresponding to DeePMD-kit's default units of eV. |

When using DeePMD-kit models, the following extra keyword arguments to `createSystem()` and `createMixedSystem()` are supported.

| Argument | Description |
| --- | --- |
| `lambdaName` | The name of a lambda parameter to use for alchemical calculations.  The default value is `None`, which does not create a lambda parameter. |
 | `lambdaValue` | The initial value of the lambda parameter |

### Other Packages

OpenMM-ML is based on a plugin architecture, allowing other packages to provide their own interfaces to it.  The
packages listed above are the ones for which OpenMM-ML has built in support.  Other packages can interface to it by
defining two classes that subclass `MLPotentialImpl` and `MLPotentialImplFactory`, then registering them by specifying
an [entry point](https://packaging.python.org/en/latest/specifications/entry-points/) in the group `openmmml.potentials`.
Consult the documentation for other packages to see whether they provide interfaces for OpenMM-ML.