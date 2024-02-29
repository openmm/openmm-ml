"""
aimnet2potential.py: Implements the AIMNet2 potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021-2023 Stanford University and the Authors.
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
from typing import Iterable, Optional, Union

class AIMNet2PotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates AIMNet2PotentialImpl objects."""

    def createImpl(self, name: str, **args) -> MLPotentialImpl:
        return AIMNet2PotentialImpl(name)


class AIMNet2PotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the AIMNet2 potential.
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

        import torch
        import openmmtorch
        if self.name == 'aimnet2-wb97m-d3':
            model = torch.load('aimnet2_wb97m-d3_ens.jpt')
        else:
            raise ValueError('Unsupported AIMNet2 model: '+self.name)

        # Create the PyTorch model that will be invoked by OpenMM.

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        numbers = torch.tensor([[atom.element.atomic_number for atom in includedAtoms]])
        charge = torch.tensor([args['charge']], dtype=torch.float32)

        class AIMNet2Force(torch.nn.Module):

            def __init__(self, model, numbers, charge, atoms):
                super(AIMNet2Force, self).__init__()
                self.model = model
                self.numbers = torch.nn.Parameter(numbers, requires_grad=False)
                self.charge = torch.nn.Parameter(charge, requires_grad=False)
                self.energyScale = (unit.ev/unit.item).conversion_factor_to(unit.kilojoules_per_mole)
                if atoms is None:
                    self.indices = None
                else:
                    self.indices = torch.tensor(sorted(atoms), dtype=torch.int64)

            def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):
                positions = positions.to(torch.float32)
                if self.indices is not None:
                    positions = positions[self.indices]
                args = {'coord': 10.0*positions.unsqueeze(0),
                        'numbers': self.numbers,
                        'charge': self.charge}
                result = self.model(args)
                return (self.energyScale*result['energy'], (10.0*self.energyScale)*result['forces'][0])

        # Create the TorchForce and add it to the System.

        module = torch.jit.script(AIMNet2Force(model, numbers, charge, atoms))
        force = openmmtorch.TorchForce(module)
        force.setForceGroup(forceGroup)
        force.setOutputsForces(True)
        system.addForce(force)

MLPotential.registerImplFactory('aimnet2-wb97m-d3', AIMNet2PotentialImplFactory())
