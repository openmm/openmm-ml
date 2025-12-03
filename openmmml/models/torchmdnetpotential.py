"""
torchmdnetpotential.py: Implements the TorchMD-Net potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021-2023 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Stephen Farr

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

class TorchMDNetPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates TorchMDNetPotentialImpl objects."""

    def createImpl(
        self, name: str, modelPath: Optional[str] = None, **args
    ) -> MLPotentialImpl:
        return TorchMDNetPotentialImpl(name, modelPath)


class TorchMDNetPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the TorchMDNet potential.

    The TorchMDNet potential is constructed using `torchmdnet` to build a PyTorch model,
    and then integrated into the OpenMM System using a TorchForce.  To use it, specify the model by name
    and provide the path to a model.

    >>> potential = MLPotential('TorchMD-Net', modelPath=<model_file_path>)

    During system creation you can enable cudagraphs for a speed-up for small molecules

    >>>  system = potential.createSystem(pdb.topology, cudaGraphs=True)

    You can also specify the molecule's total charge

    >>>  system = potential.createSystem(pdb.topology, charge=0)

    
    Attributes
    ----------
    name : str
        The name of the potential ('TorchMD-Net')
    modelPath : str
        The path to the model file
    """


    def __init__(self, name: str, modelPath) -> None:
        self.name = name
        self.modelPath = modelPath

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  **args):
        # Load the TorchMDNet model.


        try:
            from torchmdnet.models.model import load_model
        except ImportError as e:
            raise ImportError(f"Failed to import torchmdnet please install from https://torchmd-net.readthedocs.io/en/latest/installation.html")
        import torch
        import openmmtorch

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        numbers = torch.tensor([atom.element.atomic_number for atom in includedAtoms])
        charge = torch.tensor([args.get('charge', 0)], dtype=torch.float32)


        model_file_path = self.modelPath


        model = load_model(
            model_file_path,
            derivative=False,
            remove_ref_energy = args.get('remove_ref_energy', True),
            max_num_neighbors = min(args.get('max_num_neighbors', 64), numbers.shape[0]),
            static_shapes = True,
            check_errors = False
        )
        for parameter in model.parameters():
            parameter.requires_grad = False


        
        _batch = args.get('batch', None)
        if _batch is None:
            _batch = torch.zeros_like(numbers)
        else:
            assert _batch.dtype == torch.long and _batch.dim()==1

        batch = _batch

        use_cudagraphs = args.get('cudaGraphs', False)


        class TorchMDNetForce(torch.nn.Module):

            def __init__(self, model, numbers, charge, atoms, batch):
                super(TorchMDNetForce, self).__init__()
                self.model = model
                self.numbers = torch.nn.Parameter(numbers, requires_grad=False)
                self.charge = torch.nn.Parameter(charge, requires_grad=False)
                self.batch = torch.nn.Parameter(batch, requires_grad=False)
                
                if atoms is None:
                    self.subset = False
                    self.indices = torch.empty(1) # for torchscript
                else:
                    self.subset = True
                    self.indices = torch.nn.Parameter(torch.tensor(sorted(atoms), dtype=torch.int64), requires_grad=False)
                    

            def forward(self, positions: torch.Tensor, boxvectors: Optional[torch.Tensor] = None):
                positions = positions.to(torch.float32).to(self.numbers.device)
                if self.subset:
                    positions = positions[self.indices]

                energy = self.model(z=self.numbers, pos=positions*10.0, batch=self.batch, q=self.charge)[0]
                return energy * 96.4916  # eV -> kJ/mol

        # Create the TorchForce and add it to the System.
        module = torch.jit.script(TorchMDNetForce(model, numbers, charge, atoms, batch)).to(torch.device('cpu'))
        force = openmmtorch.TorchForce(module)
        if use_cudagraphs:
            force.setProperty("useCUDAGraphs", "true")
        force.setForceGroup(forceGroup)
        system.addForce(force)
