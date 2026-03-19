"""
torchmdnetpotential.py: Implements the TorchMD-Net potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021-2026 Stanford University and the Authors.
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
import openmm
from openmm import unit
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory
from typing import Iterable, Optional
import numpy as np

class TorchMDNetPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates TorchMDNetPotentialImpl objects."""

    def createImpl(
        self, 
        name: str, 
        modelPath: Optional[str] = None,
        lengthScale: float = 0.1, # angstrom -> nm
        energyScale: float = 96.4916,  # eV -> kJ/mol
    ) -> MLPotentialImpl:
        return TorchMDNetPotentialImpl(name, modelPath, lengthScale, energyScale)

class TorchMDNetPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the TorchMDNet potential.

    The TorchMDNet potential is constructed using `torchmdnet` to build a PyTorch model,
    and then integrated into the OpenMM System using a TorchForce.  To use it, specify the model by name
    and provide the path to a model.

    >>> potential = MLPotential('torchmdnet', modelPath=<model_file_path>)

    The default energy and length scales assume a model is trained with positions in angstroms and energies in eV.
    If this is not the case you can specify the length and energy scales by passing the factors that convert the model
    distance to nm and the energy to kJ/mol, for example:

    >>> potential = MLPotential('torchmdnet', modelPath=<model_file_path>, 
                                lengthScale=0.1 # angstrom to nm, 
                                energyScale=4.184 # kcal/mol to kJ/mol)

    During system creation you can enable CUDA graphs for a speed-up for small molecules:

    >>>  system = potential.createSystem(pdb.topology, cudaGraphs=True)

    The default is to enable this for TensorNet models.

    You can also specify the molecule's total charge:

    >>>  system = potential.createSystem(pdb.topology, charge=0)

    Pretained AceFF models can be used directly:

    >>> potential = MLPotential('aceff-2.0')

    >>> potential = MLPotential('aceff-1.1')

    >>> potential = MLPotential('aceff-1.0')

    """

    def __init__(self, 
                 name: str,
                 modelPath: str,
                 lengthScale: float,
                 energyScale: float
    ) -> None:
        """
        Initialize the TorchMDNetPotentialImpl.

        Parameters
        ----------
        name : str
            The name of the model.
            'torchmdnet' for a local model file, or pretrained models are available: 'aceff-1.0' or 'aceff-1.1'. 
        modelPath : str, optional
            The path to the locally trained torchmdnet model if ``name`` is 'torchmdnet'.
        lengthScale : float
            The length conversion factor from the model units to nanometers. 
            If not specified the default is 0.1 which corresponds to a model in angstrom
        energyScale : float
            The energy conversion factor from the model units to kJ/mol.
            If not specified the default is 96.4916 which corresponds to a model in eV.
   
        """
        self.name = name
        self.modelPath = modelPath
        self.lengthScale = lengthScale
        self.energyScale = energyScale

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  **args):
        # Load the TorchMDNet model.
        try:
            import torchmdnet
            from torchmdnet.models.model import load_model
        except ImportError as e:
            raise ImportError(f"Failed to import torchmdnet please install from https://torchmd-net.readthedocs.io/en/latest/installation.html")
        import torch

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        device = self._getTorchDevice(args)
        numbers = torch.tensor([atom.element.atomic_number for atom in includedAtoms], device=device, requires_grad=False)
        charge = torch.tensor([args.get('charge', 0)], dtype=torch.float32, device=device, requires_grad=False)
        cutoff = 10*args.get('coulomb_cutoff', 1.2)
        if unit.is_quantity(cutoff):
            cutoff = cutoff.value_in_unit(unit.angstrom)

        if self.name == 'torchmdnet':
            # a local path to a torchmdnet checkpoint must be provided 
            model_file_path = self.modelPath
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise ImportError(f"Failed to import huggingface_hub please install from https://huggingface.co/docs/huggingface_hub/en/installation")
            
            if self.name == 'aceff-1.0':
                repo_id="Acellera/AceFF-1.0"
                filename="aceff_v1.0.ckpt"
            elif self.name == 'aceff-1.1':
                repo_id="Acellera/AceFF-1.1"
                filename="aceff_v1.1.ckpt"
            elif self.name == 'aceff-2.0':
                repo_id="Acellera/AceFF-2.0"
                filename="aceff_v2.0.ckpt"
            else:
                raise ValueError(f'Model name {self.name} does not exist.')

            model_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )

        model = load_model(
            model_file_path,
            derivative=False,
            remove_ref_energy = args.get('remove_ref_energy', True),
            max_num_neighbors = min(args.get('max_num_neighbors', 64), numbers.shape[0]),
            coulomb_cutoff = cutoff,
            static_shapes = True,
            check_errors = False
        ).to(device)
        for parameter in model.parameters():
            parameter.requires_grad = False
        batch = args.get('batch', None)
        if batch is None:
            batch = torch.zeros_like(numbers, requires_grad=False)
        else:
            batch = torch.tensor(batch, dtype=torch.long, device=device, requires_grad=False)
        if atoms is None:
            indices = None
        else:
            indices = np.array(sorted(atoms))
        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()

        # Create the PythonForce and add it to the System.

        compute = _ComputeTorchMDNet(model=model,
                                     numbers=numbers,
                                     charge=charge,
                                     batch=batch,
                                     lengthScale=self.lengthScale,
                                     energyScale=self.energyScale,
                                     indices=indices,
                                     periodic=periodic)
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)


class _ComputeTorchMDNet(object):
    def __init__(self, model, numbers, charge, batch, lengthScale, energyScale, indices, periodic):
        self.model = model
        self.compiled_model = None
        self.numbers = numbers
        self.charge = charge
        self.batch = batch
        self.lengthScale = lengthScale
        self.energyScale = energyScale
        self.indices = indices
        self.periodic = periodic

    def __call__(self, state):
        import torch
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        numAtoms = positions.shape[0]
        positions = torch.tensor(positions, dtype=torch.float32, device=self.numbers.device)
        if self.indices is not None:
            positions = positions[self.indices]
        positions.requires_grad_(True)
        if self.periodic:
            cell = torch.tensor(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer), dtype=torch.float32, device=self.numbers.device)/self.lengthScale
        else:
            cell = None

        if self.compiled_model is None:
            # Reset dynamo caches to avoid conflicts with compiled state
            # from a previous molecule's model.
            torch._dynamo.reset()
            # Warmup pass to set dim_size before compilation.
            # torch.compile doesn't support .item() calls used internally.
            self.model.to(self.numbers.device)
            with torch.no_grad():
                self.model(z=self.numbers, pos=positions/self.lengthScale, batch=self.batch, q=self.charge, box=cell)
            self.compiled_model = torch.compile(self.model, backend="inductor", dynamic=False, fullgraph=True, mode="default")

        energy = self.compiled_model(z=self.numbers, pos=positions/self.lengthScale, batch=self.batch, q=self.charge, box=cell)[0]*self.energyScale
        energy.backward()
        forces = (-positions.grad).detach().cpu().numpy()
        if self.indices is not None:
            f = np.zeros((numAtoms, 3), dtype=np.float32)
            f[self.indices] = forces
            forces = f
        return energy, forces
