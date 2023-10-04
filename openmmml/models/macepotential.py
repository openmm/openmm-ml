"""
macepotential.py: Implements the MACE potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021 Stanford University and the Authors.
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
from typing import Iterable, Optional, Union, Tuple
from openmmml.models.utils import simple_nl
import logging


class MACEPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates MACEPotentialImpl objects."""

    def createImpl(self, name: str, model_path: str, **args) -> MLPotentialImpl:
        return MACEPotentialImpl(name, model_path)

class MACEPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the MACE potential.

    The potential is implemented using MACE to build a PyTorch model.  A
    TorchForce is used to add it to the OpenMM System.

    You must specify the path to a trained MACE model e.g.
    >>> potential = MLPotential('mace', model_path='MACE.model')

    When you create the system can optionally change the dtype of the model
    with the keyword argument dtype='float32' or dtype='float64'. The default behavoir is
    to use the dtype of the loaded MACE model.
    e.g.
    >>> system = potential.createSystem(topology, dtype='float32')

    You can also specify that the full atomic energy (including the atom self energy) is returned 
    rather than the default of the interaction energy.
    e.g.
    >>> system = potential.createSystem(topology, interaction_energy=False)
    
    """

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  implementation : str=None,
                  dtype: str=None,
                  interaction_energy: bool=True,
                  **args):
        

        import torch
        import openmmtorch
        from e3nn.util import jit
        from mace.tools import utils, to_one_hot, atomic_numbers_to_indices
        
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        
        class MACEForce(torch.nn.Module):

            def __init__(self, model_path, atomic_numbers, indices, periodic, dtype=None, interaction_energy=True):
                super(MACEForce, self).__init__()


                if not torch.cuda.is_available():
                    self.model = torch.load(model_path, map_location=torch.device('cpu'))
                else:
                    self.model = torch.load(model_path)
                
                # get the dtype of the saved model
                model_dtype = [p.dtype for p in self.model.parameters()][0]

                # check is user has request a different dtype    
                if dtype is not None and dtype != model_dtype:
                    logging.getLogger().warning(f'Warning: Loaded MACE model has dtype of {model_dtype} which is being changed to {dtype} as requested!')
                    self.default_dtype = dtype
                    self.model.to(self.default_dtype)
                else:
                    self.default_dtype = model_dtype

                torch.set_default_dtype(self.default_dtype)

                # conversion constants 
                self.register_buffer('nm_to_distance', torch.tensor(10.0)) # nm->A
                self.register_buffer('distance_to_nm', torch.tensor(0.1)) # A->nm
                self.register_buffer('energy_to_kJ', torch.tensor(96.49)) # eV->kJ


                #self.register_buffer('r_max', self.model.r_max)
                self.z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])

                self.model = jit.compile(self.model)
                
                # setup input
                N = len(atomic_numbers)
                self.ptr = torch.nn.Parameter(torch.tensor([0,N], dtype=torch.long), requires_grad=False)
                self.batch = torch.nn.Parameter(torch.zeros(N, dtype=torch.long), requires_grad=False)
                
                # one hot encoding of atomic number
                self.node_attrs = torch.nn.Parameter(to_one_hot(
                        torch.tensor(atomic_numbers_to_indices(atomic_numbers, z_table=self.z_table), dtype=torch.long).unsqueeze(-1),
                        num_classes=len(self.z_table),
                    ), requires_grad=False)

                if periodic:
                    self.pbc = torch.nn.Parameter(torch.tensor([True, True, True]), requires_grad=False)
                else:
                    self.pbc = torch.nn.Parameter(torch.tensor([False, False, False]), requires_grad=False)


                if indices is None:
                    self.indices = None
                else:
                    self.indices = torch.tensor(indices, dtype=torch.int64)

                if interaction_energy is True:
                    self.return_energy_type = "interaction_energy"
                else:
                    self.return_energy_type = "energy"

            def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):
                
                # setup positions
                positions = positions.to(self.default_dtype)
                if self.indices is not None:
                    positions = positions[self.indices]

                positions = positions*self.nm_to_distance

                if boxvectors is not None:
                    cell = boxvectors.to(self.default_dtype) * self.nm_to_distance
                    pbc = True
                else:
                    cell = torch.eye(3, device=positions.device)
                    pbc = False
  
                mapping, shifts_idx = simple_nl(positions, cell, pbc, self.model.r_max)
                
                edge_index = torch.stack((mapping[0], mapping[1]))

                shifts = torch.mm(shifts_idx, cell)

                # create input dict
                input_dict = { "ptr" : self.ptr, 
                              "node_attrs": self.node_attrs, 
                              "batch": self.batch, 
                              "pbc": self.pbc,
                              "cell": cell,
                              "positions": positions,
                              "edge_index": edge_index,
                              "unit_shifts": shifts_idx,
                              "shifts": shifts}
                
                # predict
                out = self.model(input_dict,compute_force=False)

                energy = out[self.return_energy_type]
                if energy is None:
                    energy = torch.tensor(0.0, dtype=self.default_dtype, device=positions.device)
                
                # return energy 
                energy = energy*self.energy_to_kJ

                return energy
            

        is_periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()


        atomic_numbers = [atom.element.atomic_number for atom in includedAtoms]

        if dtype is None:
            torch_dtype = None
        elif dtype in ['float32', 'float64']:
            torch_dtype = {'float32':torch.float32, 'float64':torch.float64}[dtype]
        else:
            raise ValueError(f'Specified dtype of {dtype} is not valid. Allowed values are None, "float32" or "float64"')

        maceforce = MACEForce(self.model_path, atomic_numbers, atoms, is_periodic, dtype=torch_dtype, interaction_energy=interaction_energy)
        
        # Convert it to TorchScript
        module = torch.jit.script(maceforce)

        # Create the TorchForce and add it to the System.
        force = openmmtorch.TorchForce(module)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(is_periodic)
        system.addForce(force)

MLPotential.registerImplFactory('mace', MACEPotentialImplFactory())
