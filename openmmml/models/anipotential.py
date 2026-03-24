"""
anipotential.py: Implements the ANI potential function using TorchANI.

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

class ANIPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates ANIPotentialImpl objects."""

    def createImpl(self, name: str, **args) -> MLPotentialImpl:
        return ANIPotentialImpl(name)


class ANIPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the ANI potential.

    The potential is implemented using TorchANI to build a PyTorch model.  A
    TorchForce is used to add it to the OpenMM System.  The ANI1ccx and ANI2x
    versions are currently supported.

    Both ANI1ccx and ANI2x are ensembles of eight models.  Averaging over all eight
    models leads to slightly more accurate results than any one individually.  You
    can optionally use only a single model by specifying the modelIndex argument to
    select which one to use.  This leads to a large improvement in speed, at the
    cost of a small decrease in accuracy.
    """

    def __init__(self, name):
        self.name = name


    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  modelIndex: Optional[int] = None,
                  **args):
        # Create the TorchANI model.

        try:
            import torchani
        except ImportError as e:
            raise ImportError(f"Failed to import torchani with error: {e}. Make sure torchani is installed.")
        import torch
        import numpy as np
        device = self._getTorchDevice(args)
        _kwarg_dict = {'periodic_table_index': True}
        if self.name == 'ani1ccx':
            model = torchani.models.ANI1ccx(**_kwarg_dict)
        elif self.name == 'ani2x':
            model = torchani.models.ANI2x(**_kwarg_dict)
        else:
            raise ValueError('Unsupported ANI model: '+self.name)
        if modelIndex is not None:
            model = model[modelIndex]
        model.to(device)

        # Prepare inputs to the model.

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        species = torch.tensor([[atom.element.atomic_number for atom in includedAtoms]], device=device)
        if atoms is None:
            indices = None
        else:
            indices = np.array(atoms)
        periodic = topology.getPeriodicBoxVectors() is not None or system.usesPeriodicBoundaryConditions()
        if periodic:
            pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
        else:
            pbc = None

        # Create the PythonForce and add it to the System.

        compute = partial(_computeANI,
                          model=model,
                          species=species,
                          pbc=pbc,
                          indices=indices)
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)

def _computeANI(state, model, species, pbc, indices):
    import torch
    import numpy as np
    import torchani
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    numAtoms = positions.shape[0]
    if indices is not None:
        positions = positions[indices]
    positions = torch.tensor(positions, dtype=torch.float32, device=species.device)
    if pbc is None:
        boxvectors = None
    else:
        boxvectors = torch.tensor(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom), dtype=torch.float32, device=species.device)
        positions = positions - torch.outer(torch.floor(positions[:,2]/boxvectors[2,2]), boxvectors[2])
        positions = positions - torch.outer(torch.floor(positions[:,1]/boxvectors[1,1]), boxvectors[1])
        positions = positions - torch.outer(torch.floor(positions[:,0]/boxvectors[0,0]), boxvectors[0])
    positions = positions.unsqueeze(0)
    positions.requires_grad_(True)
    _, energy = model((species, positions), cell=boxvectors, pbc=pbc)
    energy *= torchani.units.hartree2kjoulemol(1)
    energy.backward()
    forces = (-positions.grad).detach().cpu().numpy()
    if indices is not None:
        f = np.zeros((numAtoms, 3), dtype=np.float32)
        f[indices] = forces
        forces = f
    return energy, forces
