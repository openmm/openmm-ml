"""
aimnet2potential.py: Implements the AIMNet2 potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021-2026 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

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

from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
import openmm
from openmm import unit
from typing import Iterable, Optional
from functools import partial
import numpy as np

class AIMNet2PotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates AIMNet2PotentialImpl objects."""

    def createImpl(self, name: str, modelPath: Optional[str] = None, **args) -> MLPotentialImpl:
        return AIMNet2PotentialImpl(name, modelPath)


class AIMNet2PotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the AIMNet2 potential.

    The AIMNet2 potential is constructed using `aimnet` to build a PyTorch model,
    and then integrated into the OpenMM System using a PythonForce.  This
    implementation supports both the pretrained AIMNet2 models and locally trained
    AIMNet2 models.

    To use a pretrained model, specify a supported AIMNet2 model name.  The supported
    model families are 'aimnet2' (general organic/elemental-organic), 'aimnet2-2025'
    (improved intermolecular interactions), 'aimnet2-nse' (open-shell systems and
    radicals), 'aimnet2-pd' (Pd catalysis), and 'aimnet2-rxn' (reactive chemistry).  For
    example:

    >>> potential = MLPotential('aimnet2')

    Each family is a four-member ensemble.  Pass the ``modelIndex`` argument (0 through 3)
    to ``createSystem()`` to select which member to use.  If it is omitted, member 0 is
    used by default:

    >>> system = potential.createSystem(topology, modelIndex=2)

    To use a locally trained AIMNet2 model, use the name 'aimnet' and provide the
    path to the model file:

    >>> potential = MLPotential('aimnet', modelPath='mymodel.pt')

    Attributes
    ----------
    name : str
        The name of the AIMNet2 model.  This is either an AIMNet2 registry model name
        or alias for a pretrained model, or 'aimnet' for a locally trained model.
    modelPath : str
        The path to the locally trained AIMNet2 model if ``name`` is 'aimnet'.
    """

    # (Family alias, canonical AIMNet2 registry family prefix)
    # Model index (0-3) is appended at load time, e.g. 'aimnet2-wb97m-d3_2'.
    KNOWN_MODELS = {
        'aimnet2':      'aimnet2-wb97m-d3',
        'aimnet2-2025': 'aimnet2-b973c-2025-d3',
        'aimnet2-nse':  'aimnet2-nse',
        'aimnet2-pd':   'aimnet2-pd',
        'aimnet2-rxn':  'aimnet2-rxn',
    }

    def __init__(self, name, modelPath=None):
        self.name = name
        self.modelPath = modelPath

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  modelIndex: int = 0,
                  **args):
        # Load the AIMNet2 model.

        try:
            from aimnet.calculators import AIMNet2Calculator
        except ImportError as e:
            raise ImportError(f"Failed to import aimnet with error: {e}. Install from https://github.com/isayevlab/aimnetcentral.")
        import torch
        if self.name in AIMNet2PotentialImpl.KNOWN_MODELS:
            # We select an ensemble member by substituting its index into the family's
            # canonical member name (e.g. 'aimnet2-wb97m-d3_2'), using the prefixes in
            # KNOWN_MODELS. We can't instead pass the family alias with
            # AIMNet2Calculator's `ensemble_member` argument: for a registry alias that
            # argument is silently ignored (it only applies when loading an ensemble from
            # a HuggingFace repo), so every index would return member 0 and out-of-range
            # values would raise nothing.
            if not 0 <= modelIndex <= 3:
                raise ValueError(f"modelIndex must be 0-3 for {self.name}, got {modelIndex}")
            modelName = f'{AIMNet2PotentialImpl.KNOWN_MODELS[self.name]}_{modelIndex}'
            model = AIMNet2Calculator(modelName)
        elif self.name == 'aimnet':
            if self.modelPath is None:
                raise ValueError("No modelPath provided for local AIMNet2 model.")
            if modelIndex != 0:
                raise ValueError("modelIndex != 0 is not supported for local AIMNet2 models "
                                 "('aimnet') -- only the model at modelPath is available.")
            model = AIMNet2Calculator(self.modelPath)
        else:
            raise ValueError(f"Unsupported AIMNet2 model: {self.name}")
        device = torch.device(model.device)
        model.device = device

        # Create the PyTorch model that will be invoked by OpenMM.

        includedAtoms = list(topology.atoms())
        if atoms is None:
            indices = None
        else:
            includedAtoms = [includedAtoms[i] for i in atoms]
            indices = np.array(atoms)
        numbers = torch.tensor([[atom.element.atomic_number for atom in includedAtoms]], device=device)
        charge = torch.tensor([args.get('charge', 0)], dtype=torch.float32, device=device)
        multiplicity = torch.tensor([args.get('multiplicity', 1)], dtype=torch.float32, device=device)
        periodic = topology.getPeriodicBoxVectors() is not None

        # Create the PythonForce and add it to the System.

        compute = partial(_computeAIMNet2, model=model, numbers=numbers, charge=charge, multiplicity=multiplicity, indices=indices, periodic=periodic)
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)

    def getMLLongRange(self) -> bool | None:
        # NOTE: By default, creating an AIMNet2Calculator gives a coulomb_method
        # of "simple" (which changes to the cutoff-based "dsf" method with PBCs;
        # see https://isayevlab.github.io/aimnetcentral/long_range/).  This is a
        # property of the calculator rather than the individual model weights, so it
        # is assumed to hold for every supported registry family as well as for a
        # user-supplied local model (modelPath).  OpenMM-ML does not expose any
        # option to change this; if we change this behavior in the future, or a
        # supported model has different behavior, this must be updated.
        return False

def _computeAIMNet2(state, model, numbers, charge, multiplicity, indices, periodic):
    import torch
    positions = torch.tensor(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom), dtype=torch.float32, device=numbers.device)
    numAtoms = positions.shape[0]
    if indices is not None:
        positions = positions[indices]
    args = {'coord': positions.unsqueeze(0),
            'numbers': numbers,
            'charge': charge,
            'mult': multiplicity}
    if periodic:
        cell = torch.tensor(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom), dtype=torch.float32, device=numbers.device)
        args['cell'] = cell
    result = model(args, forces=True)
    energyScale = (unit.ev/unit.item).conversion_factor_to(unit.kilojoules_per_mole)
    energy = float(energyScale*result["energy"].sum().detach())
    forces = (10.0*energyScale*result["forces"]).detach().cpu().numpy()[0]
    if indices is not None:
        f = np.zeros((numAtoms, 3), dtype=np.float32)
        f[indices] = forces
        forces = f
    return energy, forces
