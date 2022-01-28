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
from typing import Iterable, Optional

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

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  filename: str = 'animodel.pt',
                  use_OptimizedTorchANI : Optional[bool] = True, 
                  **args):
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
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        elements = [atom.element.symbol for atom in includedAtoms]
        species = model.species_to_tensor(elements).unsqueeze(0)
        
        if use_OptimizedTorchANI: # ask to optimize torchani
            try:
                # from https://github.com/openmm/NNPOps#example
                from NNPOps import OptimizedTorchANI
                model = OptimizedTorchANI(model, species)
            except Exception as e:
                raise Exception(e)
                
        class ANIForce(torch.nn.Module):

            def __init__(self, model, species, atoms, periodic):
                super(ANIForce, self).__init__()
                self.model = model
                self.species = species
                self.energyScale = torchani.units.hartree2kjoulemol(1)
                if atoms is None:
                    self.indices = None
                else:
                    self.indices = torch.tensor(sorted(atoms), dtype=torch.int64)
                if periodic:
                    self.pbc = torch.tensor([True, True, True], dtype=torch.bool)
                else:
                    self.pbc = None

            def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):
                positions = positions.to(torch.float32)
                if self.indices is not None:
                    positions = positions[self.indices]
                if boxvectors is None:
                    _, energy = self.model((self.species, 10.0*positions.unsqueeze(0)))
                else:
                    boxvectors = boxvectors.to(torch.float32)
                    _, energy = self.model((self.species, 10.0*positions.unsqueeze(0)), cell=10.0*boxvectors, pbc=self.pbc)
                return self.energyScale*energy

        aniForce = ANIForce(model, species, atoms, topology.getPeriodicBoxVectors() is not None)

        # Convert it to TorchScript and save it.

        module = torch.jit.script(aniForce)
        module.save(filename)

        # Create the TorchForce and add it to the System.

        force = openmmtorch.TorchForce(filename)
        force.setForceGroup(forceGroup)
        if topology.getPeriodicBoxVectors() is not None:
            force.setUsesPeriodicBoundaryConditions(True)
        system.addForce(force)

MLPotential.registerImplFactory('ani1ccx', ANIPotentialImplFactory())
MLPotential.registerImplFactory('ani2x', ANIPotentialImplFactory())
