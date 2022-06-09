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
from typing import Iterable, Optional, Union

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
                  implementation : str = 'nnpops',
                  overwrite_periodicity : Optional[bool]=False,
                  **args):
        """
        overwrite_periodicity : whether to overwrite the periodicity given by `topology` or `system` to its opposite; this is primarily for 
            testing purposes
        """
        # Create the TorchANI model.

        import torchani
        import torch
        import openmmtorch

        # `nnpops` throws error if `periodic_table_index`=False if one passes `species` as `species_to_tensor` from `element`
        _kwarg_dict = {'periodic_table_index': True} #if implementation == 'nnpops' else {}
        if self.name == 'ani1ccx':
            model = torchani.models.ANI1ccx(**_kwarg_dict)
        elif self.name == 'ani2x':
            model = torchani.models.ANI2x(**_kwarg_dict)
        else:
            raise ValueError('Unsupported ANI model: '+self.name)

        # Create the PyTorch model that will be invoked by OpenMM.

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        # elements = [atom.element.symbol for atom in includedAtoms]
        #species = model.species_to_tensor(elements).unsqueeze(0)
        species = torch.tensor([[atom.element.atomic_number for atom in includedAtoms]])

        if implementation == 'nnpops':
            try:
                from NNPOps import OptimizedTorchANI
                device = torch.device('cuda') # nnpops needs cuda
            except Exception as e:
                print(f"failed to import `nnpops` with error: {e}")
            #species = torch.tensor([[atom.element.atomic_number for atom in includedAtoms]], device=device) # with nnpops, one needs `atomic_number`
            model = OptimizedTorchANI(model, species).to(device)
        elif implementation == "torchani":
            pass # do nothing
        else:
            raise NotImplementedError(f"implementation {implementation} is not supported")

        class ANIForce(torch.nn.Module):

            def __init__(self, model, species, atoms, periodic):
                super(ANIForce, self).__init__()
                self.model = model
                self.species = torch.nn.Parameter(species, requires_grad=False)
                self.energyScale = torchani.units.hartree2kjoulemol(1)
                if atoms is None:
                    self.indices = None
                else:
                    self.indices = torch.tensor(sorted(atoms), dtype=torch.int64)
                if periodic:
                    self.pbc = torch.nn.Parameter(torch.tensor([True, True, True], dtype=torch.bool), requires_grad=False)
                else:
                    self.pbc = None

            def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):
                positions = positions.to(torch.float32)
                #print(f"(boxvectors, scale): {boxvectors, scale}")
                if self.indices is not None:
                    positions = positions[self.indices]
                if boxvectors is None:
                    _, energy = self.model((self.species, 10.0*positions.unsqueeze(0)))
                else:
                    boxvectors = boxvectors.to(torch.float32)
                    _, energy = self.model((self.species, 10.0*positions.unsqueeze(0)), cell=10.0*boxvectors, pbc=self.pbc)

                return self.energyScale*energy

        # is_periodic...
        is_periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()
        if overwrite_periodicity: # flip the periodicity bool
            is_periodic = not is_periodic
        aniForce = ANIForce(model, species, atoms, is_periodic)

        # Convert it to TorchScript and save it.

        module = torch.jit.script(aniForce)
        module.save(filename)

        # Create the TorchForce and add it to the System.

        force = openmmtorch.TorchForce(filename)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(is_periodic)
        if is_periodic:
            print(f"using pbcs...")
        else:
            print(f"omitting pbcs since the querying the topology and system gave is_periodic={is_periodic}")
        system.addForce(force)

MLPotential.registerImplFactory('ani1ccx', ANIPotentialImplFactory())
MLPotential.registerImplFactory('ani2x', ANIPotentialImplFactory())
