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

class AIMNet2PotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates AIMNet2PotentialImpl objects."""

    def createImpl(self, name: str, **args) -> MLPotentialImpl:
        return AIMNet2PotentialImpl(name)


class AIMNet2PotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the AIMNet2 potential.

    The AIMNet2 potential is constructed using `aimnet` to build a PyTorch model,
    and then integrated into the OpenMM System using a TorchForce.  To use it, specify the model by name:

    >>> potential = MLPotential('aimnet2')
    """

    def __init__(self, name):
        self.name = name

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  **args):
        # Load the AIMNet2 model.

        try:
            from aimnet.calculators import AIMNet2Calculator
        except ImportError as e:
            raise ImportError(f"Failed to import aimnet with error: {e}. Install from https://github.com/isayevlab/aimnetcentral.")
        import torch
        model = AIMNet2Calculator('aimnet2')
        device = torch.device(model.device)
        model.device = device

        # Create the PyTorch model that will be invoked by OpenMM.

        includedAtoms = list(topology.atoms())
        if atoms is None:
            indices = None
        else:
            includedAtoms = [includedAtoms[i] for i in atoms]
            indices = torch.tensor(sorted(atoms), dtype=torch.int64, device=device)
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


def _computeAIMNet2(state, model, numbers, charge, multiplicity, indices, periodic):
    import torch
    import numpy as np
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
