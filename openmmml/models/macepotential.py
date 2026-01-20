"""
macepotential.py: Implements the MACE potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021-2026 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Stephen Farr, Joao Morado

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import openmm
from openmm import unit
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
from typing import Iterable, Optional
from functools import partial


class MACEPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates MACEPotentialImpl objects."""

    def createImpl(
        self, name: str, modelPath: Optional[str] = None, **args
    ) -> MLPotentialImpl:
        return MACEPotentialImpl(name, modelPath)


class MACEPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the MACE potential.

    The MACE potential is constructed using MACE to build a PyTorch model,
    and then integrated into the OpenMM System using a TorchForce.
    This implementation supports both foundation models and locally trained MACE models.

    To use one of the pre-trained MACE foundation models, specify the model name. For example:

    >>> potential = MLPotential('mace-off23-small')

    Other available models include 'mace-off23-medium', 'mace-off23-large', 'mace-off24-medium',
    'mace-mpa-0-medium', 'mace-omat-0-small', 'mace-omat-0-medium', and 'mace-omol-0-extra-large'.

    To use a locally trained MACE model, provide the path to the model file. For example:

    >>> potential = MLPotential('mace', modelPath='MACE.model')

    During system creation, you can optionally specify the precision of the model using the
    ``precision`` keyword argument. Supported options are 'single' and 'double'. For example:

    >>> system = potential.createSystem(topology, precision='single')

    By default, the implementation uses the precision of the loaded MACE model.
    According to the MACE documentation, 'single' precision is recommended for MD (faster but
    less accurate), while 'double' precision is recommended for geometry optimization.

    Additionally, you can request computation of the full atomic energy, including the atom
    self-energy, instead of the default interaction energy, by setting ``returnEnergyType`` to
    'energy'. For example:
    
    >>> system = potential.createSystem(topology, returnEnergyType='energy')

    The default is to compute the interaction energy, which can be made explicit by setting
    ``returnEnergyType='interaction_energy'``.

    Attributes
    ----------
    name : str
        The name of the MACE model.
    modelPath : str
        The path to the locally trained MACE model if ``name`` is 'mace'.
    """

    def __init__(self, name: str, modelPath) -> None:
        """
        Initialize the MACEPotentialImpl.

        Parameters
        ----------
        name : str
            The name of the MACE model.
            Options include 'mace-off23-small', 'mace-off23-medium', 'mace-off23-large',
            'mace-off24-medium', 'mace-mpa-0-medium', 'mace-omat-0-small', 'mace-omat-0-medium',
            'mace-omol-0-extra-large', and 'mace'.
        modelPath : str, optional
            The path to the locally trained MACE model if ``name`` is 'mace'.
        """
        self.name = name
        self.modelPath = modelPath

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        precision: Optional[str] = None,
        returnEnergyType: str = "interaction_energy",
        **args,
    ) -> None:
        """
        Add the MACEForce to the OpenMM System.

        Parameters
        ----------
        topology : openmm.app.Topology
            The topology of the system.
        system : openmm.System
            The system to which the force will be added.
        atoms : iterable of int
            The indices of the atoms to include in the model. If ``None``, all atoms are included.
        forceGroup : int
            The force group to which the force should be assigned.
        precision : str, optional
            The precision of the model. Supported options are 'single' and 'double'.
            If ``None``, the default precision of the model is used.
        returnEnergyType : str, optional
            Whether to return the interaction energy or the energy including the self-energy.
            Default is 'interaction_energy'. Supported options are 'interaction_energy' and 'energy'.
        """
        import torch
        try:
            from mace.tools import utils, to_one_hot, atomic_numbers_to_indices
            from mace.calculators.foundations_models import mace_off, mace_mp, mace_omol
        except ImportError as e:
            raise ImportError(f"Failed to import mace with error: {e}. Install mace with 'pip install mace-torch'.")
        try:
            from e3nn.util import jit
        except ImportError as e:
            raise ImportError(f"Failed to import e3nn with error: {e}. Install e3nn with 'pip install e3nn'.")

        assert returnEnergyType in ["interaction_energy", "energy"], f"Unsupported returnEnergyType: '{returnEnergyType}'. Supported options are 'interaction_energy' or 'energy'."

        # Load the model.

        models = {
            'mace-off23-small': (mace_off, 'small', True),
            'mace-off23-medium': (mace_off, 'medium', True),
            'mace-off23-large': (mace_off, 'large', True),
            'mace-off24-medium': (mace_off, 'https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true', True),
            'mace-mpa-0-medium': (mace_mp, 'medium-mpa-0', False),
            'mace-omat-0-small': (mace_mp, 'small-omat-0', True),
            'mace-omat-0-medium': (mace_mp, 'medium-omat-0', True),
            'mace-omol-0-extra-large': (mace_omol, 'extra_large', True)
        }
        device = self._getTorchDevice(args)
        if self.name in models:
            fn, name, warn = models[self.name]
            model = fn(model=name, device=device, return_raw_model=True)
            if warn:
                import logging
                logging.warning(f'The model {self.name} is distributed under the restrictive ASL license.  Commercial use is not permitted.')
        elif self.name == "mace":
            if self.modelPath is not None:
                model = torch.load(self.modelPath, map_location="cpu")
            else:
                raise ValueError("No modelPath provided for local MACE model.")
        else:
            raise ValueError(f"Unsupported MACE model: {self.name}")

        # Get the atomic numbers of the ML region.

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        atomicNumbers = [atom.element.atomic_number for atom in includedAtoms]

        # Set the precision that the model will be used with.

        modelDefaultDtype = next(model.parameters()).dtype
        if precision is None:
            dtype = modelDefaultDtype
        elif precision == "single":
            dtype = torch.float32
        elif precision == "double":
            dtype = torch.float64
        else:
            raise ValueError(f"Unsupported precision {precision} for the model. Supported values are 'single' and 'double'.")
        if dtype != modelDefaultDtype:
            print(f"Model dtype is {modelDefaultDtype} and requested dtype is {dtype}. The model will be converted to the requested dtype.")

        # One hot encoding of atomic numbers

        zTable = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        nodeAttrs = to_one_hot(
            torch.tensor(atomic_numbers_to_indices(atomicNumbers, z_table=zTable), dtype=torch.long, device=device).unsqueeze(-1),
            num_classes=len(zTable))

        if atoms is None:
            indices = None
        else:
            indices = torch.tensor(sorted(atoms), dtype=torch.int64)
        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()

        # Create the PythonForce and add it to the System.

        compute = partial(_computeMACE,
                          model=model,
                          ptr=torch.tensor([0, nodeAttrs.shape[0]], dtype=torch.long, device=device, requires_grad=False),
                          node_attrs=nodeAttrs.to(dtype),
                          batch=torch.zeros(nodeAttrs.shape[0], dtype=torch.long, device=device, requires_grad=False),
                          pbc=torch.tensor([periodic, periodic, periodic], dtype=torch.bool, device=device, requires_grad=False),
                          returnEnergyType=returnEnergyType,
                          charge=torch.tensor([float(args.get('charge', 0))], dtype=dtype, device=device, requires_grad=False),
                          multiplicity=torch.tensor([float(args.get('multiplicity', 1))], dtype=dtype, device=device, requires_grad=False),
                          indices=indices,
                          periodic=periodic)
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)


def _computeMACE(state, model, ptr, node_attrs, batch, pbc, returnEnergyType, charge, multiplicity, indices, periodic):
    import torch
    import numpy as np
    from mace.data.neighborhood import get_neighborhood
    energyScale = 96.4853
    lengthScale = 10.0
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    numAtoms = positions.shape[0]
    if indices is not None:
        positions = positions[indices]
    if periodic:
        cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    else:
        cell = np.identity(3, dtype=np.float64)
    dtype = node_attrs.dtype
    cutoff = float(model.r_max.detach())
    edgeIndex, shifts, _, _ = get_neighborhood(positions, cutoff, [periodic, periodic, periodic], cell)
    inputDict = {
        "ptr": ptr,
        "node_attrs": node_attrs,
        "batch": batch,
        "pbc": pbc,
        "positions": torch.tensor(positions, dtype=dtype, device=ptr.device),
        "edge_index": torch.tensor(edgeIndex, dtype=torch.int64, device=ptr.device),
        "shifts": torch.tensor(shifts, dtype=dtype, device=ptr.device),
        "cell": torch.tensor(cell, dtype=dtype, device=ptr.device),
        "total_charge": charge,
        "total_spin": multiplicity
    }
    results = model(inputDict, compute_force=True)
    energy = float(results[returnEnergyType].detach())*energyScale
    forces = (results["forces"]*energyScale*lengthScale).detach().cpu().numpy()
    if indices is not None:
        f = np.zeros((numAtoms, 3), dtype=(np.float64 if dtype == torch.float64 else np.float32))
        f[indices] = forces
        forces = f
    return energy, forces
