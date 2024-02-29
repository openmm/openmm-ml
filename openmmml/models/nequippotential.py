"""
nequippotential.py: Implements the NequIP potential function.

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

class NequIPPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates NequipPotentialImpl objects."""

    def createImpl(self, name: str, model_path: str, distance_to_nm: float, energy_to_kJ_per_mol: float, atom_types: Optional[Iterable[int]]=None, **args) -> MLPotentialImpl:
        return NequIPPotentialImpl(name, model_path, distance_to_nm, energy_to_kJ_per_mol, atom_types)

class NequIPPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the NequIP potential.

    The potential is implemented using NequIP to build a PyTorch model.
    A TorchForce is used to add it to the OpenMM System. Note that you must
    provide a deployed model. No general purpose model is available.

    There are three required keyword arguments

    model_path: str
        path to deployed NequIP model
    distance_to_nm: float
        conversion constant between the nequip model distance units and OpenMM units (nm)
    energy_to_kJ_per_mol: float
        conversion constant between the nequip model energy units and OpenMM units (kJ/mol)

    for example

    >>> potential = MLPotential('nequip', model_path='example_model_deployed.pth',
                        distance_to_nm=0.1, energy_to_kJ_per_mol=4.184)    
    
    There is one optional keyword argument that lets you specify the nequip atom type of 
    each atom. Note that by default this potential uses the atomic number to map the NequIP atom type. 
    This will work if you trained your NequIP model using the standard `chemical_symbols` option. If you
    require more flexibility you can use the atom_types argument. It must be a list containing an 
    integer specifying the nequip atom type of each particle in the system.

    atom_types: List[int]


    """

    def __init__(self, name, model_path, distance_to_nm, energy_to_kJ_per_mol, atom_types):
        self.name = name
        self.model_path = model_path
        self.atom_types = atom_types
        self.distance_to_nm = distance_to_nm
        self.energy_to_kJ_per_mol = energy_to_kJ_per_mol

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  **args):
        

        import torch
        import openmmtorch
        import nequip._version
        import nequip.scripts.deploy

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        
        class NequIPForce(torch.nn.Module):

            def __init__(self, model_path, includedAtoms, indices, periodic, distance_to_nm, energy_to_kJ_per_mol, atom_types=None, verbose=None):
                super(NequIPForce, self).__init__()
                
                # conversion constants 
                self.register_buffer('nm_to_distance', torch.tensor(1.0/distance_to_nm))
                self.register_buffer('distance_to_nm', torch.tensor(distance_to_nm))
                self.register_buffer('energy_to_kJ', torch.tensor(energy_to_kJ_per_mol))

                self.model, metadata = nequip.scripts.deploy.load_deployed_model(model_path, freeze=False)

                self.default_dtype= {"float32": torch.float32, "float64": torch.float64}[metadata["model_dtype"]]
                torch.set_default_dtype(self.default_dtype)

                self.register_buffer('r_max', torch.tensor(float(metadata["r_max"])))
                
                if atom_types is not None: # use user set explicit atom types
                    nequip_types = atom_types
                
                else: # use openmm atomic symbols

                    type_names = str(metadata["type_names"]).split(" ")

                    type_name_to_type_index={ type_name : i for i,type_name in enumerate(type_names)}

                    nequip_types = [ type_name_to_type_index[atom.element.symbol] for atom in includedAtoms]
                
                atomic_numbers = [atom.element.atomic_number for atom in includedAtoms]

                self.atomic_numbers = torch.nn.Parameter(torch.tensor(atomic_numbers, dtype=torch.long), requires_grad=False)
                #self.N = len(includedAtoms)
                self.atom_types = torch.nn.Parameter(torch.tensor(nequip_types, dtype=torch.long), requires_grad=False)

                if periodic:
                    self.pbc = torch.nn.Parameter(torch.tensor([True, True, True]), requires_grad=False)
                else:
                    self.pbc = torch.nn.Parameter(torch.tensor([False, False, False]), requires_grad=False)

                # indices for ML atoms in a mixed system
                if indices is None: # default all atoms are ML
                    self.indices = None
                else:
                    self.indices = torch.tensor(indices, dtype=torch.int64)


            def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):
                
                # setup positions
                positions = positions.to(dtype=self.default_dtype)
                if self.indices is not None:
                    positions = positions[self.indices]
                positions = positions*self.nm_to_distance


                # prepare input dict 
                input_dict={}

                if boxvectors is not None:
                    input_dict["cell"]=boxvectors.to(dtype=self.default_dtype) * self.nm_to_distance
                    pbc = True
                else:
                    input_dict["cell"]=torch.eye(3, device=positions.device)
                    pbc = False

                input_dict["pbc"]=self.pbc
                input_dict["atomic_numbers"] = self.atomic_numbers
                input_dict["atom_types"] = self.atom_types
                input_dict["pos"] = positions

                # compute edges
                mapping, shifts_idx = simple_nl(positions, input_dict["cell"], pbc, self.r_max)

                input_dict["edge_index"] = mapping
                input_dict["edge_cell_shift"] = shifts_idx

                out = self.model(input_dict)    

                # return energy and forces
                energy = out["total_energy"]*self.energy_to_kJ
                forces = out["forces"]*self.energy_to_kJ/self.distance_to_nm

                return (energy, forces)
            

        is_periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()

        nequipforce = NequIPForce(self.model_path, includedAtoms, atoms, is_periodic, self.distance_to_nm, self.energy_to_kJ_per_mol, self.atom_types, **args)
        
        # Convert it to TorchScript 
        module = torch.jit.script(nequipforce)

        # Create the TorchForce and add it to the System.
        force = openmmtorch.TorchForce(module)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(is_periodic)
        force.setOutputsForces(True)
        system.addForce(force)

MLPotential.registerImplFactory('nequip', NequIPPotentialImplFactory())
