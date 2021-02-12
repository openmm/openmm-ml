"""
anipotential.py: Implements the ANI potential function using TorchANI.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021 Stanford University and the Authors.
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

class ANIPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates ANIPotentialImpl objects."""

    def createImpl(self, name: str, **args) -> MLPotentialImpl:
        return ANIPotentialImpl(name)


class ANIPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the ANI potential.

    The potential is implemented using TorchANI to build a PyTorch model.  A
    TorchForce is used to add it to the OpenMM System.  The ANI1ccx and ANI2x
    versions are currently supported.

    TorchForce requires the model to be saved to disk in a separate file.  By default
    it writes a file called 'animodel.pt' in the current working directory.  You can
    use the filename argument to specify a different name.  For example,

    >>> system = potential.createSystem(topology, filename='mymodel.pt')
    """

    def __init__(self, name):
        self.name = name

    def addForces(self, topology: openmm.app.Topology, system: openmm.System, filename='animodel.pt', **args):
        # Create the TorchANI model.

        import torchani
        import torch
        import openmmtorch
        if self.name == 'ani1ccx':
            model = torchani.models.ANI1ccx()
        elif self.name == 'ani2x':
            model = torchani.models.ANI2x()
        else:
            raise ValueError('Unsupported ANI model: '+self.name)

        # Create the PyTorch model that will be invoked by OpenMM.

        elements = ''.join([atom.element.symbol for atom in topology.atoms()])
        species = model.species_to_tensor(elements).unsqueeze(0)

        class ANIForce(torch.nn.Module):

            def __init__(self, model, species):
                super(ANIForce, self).__init__()
                self.model = model
                self.species = species

            def forward(self, positions):
                _, energy = self.model((self.species, 10.0*positions.unsqueeze(0)))
                return 2625.5000885335317*energy

        class ANIForcePeriodic(torch.nn.Module):

            def __init__(self, model, species):
                super(ANIForcePeriodic, self).__init__()
                self.model = model
                self.species = species
                self.pbc = torch.tensor([True, True, True], dtype=torch.bool)

            def forward(self, positions, boxvectors):
                _, energy = self.model((self.species, 10.0*positions.unsqueeze(0)), cell=10.0*boxvectors, pbc=self.pbc)
                return 2625.5000885335317*energy

        if topology.getPeriodicBoxVectors() is None:
            force = ANIForce(model, species)
        else:
            force = ANIForcePeriodic(model, species)

        # Convert it to TorchScript and save it.

        module = torch.jit.script(force)
        module.save(filename)

        # Create the TorchForce and add it to the System.

        force = openmmtorch.TorchForce(filename)
        if topology.getPeriodicBoxVectors() is not None:
            force.setUsesPeriodicBoundaryConditions(True)
        system.addForce(force)

MLPotential.registerImplFactory('ani1ccx', ANIPotentialImplFactory())
MLPotential.registerImplFactory('ani2x', ANIPotentialImplFactory())
