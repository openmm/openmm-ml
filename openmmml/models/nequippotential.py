"""
nequip.py: Implements the NequIP potential function.

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
from typing import Iterable, Optional, Union

class NequIPPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates NequipPotentialImpl objects."""

    def createImpl(self, name: str, model_path: str, atom_type_to_atomic_number: dict, **args) -> MLPotentialImpl:
        return NequIPPotentialImpl(name, model_path, atom_type_to_atomic_number)

class NequIPPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the NequIP potential.

    The potential is implemented using NequIP to build a PyTorch model.  A
    TorchForce is used to add it to the OpenMM System.  

    TorchForce requires the model to be saved to disk in a separate file.  By default
    it writes a file called 'nequipmodel.pt' in the current working directory.  You can
    use the filename argument to specify a different name.  For example,

    >>> system = potential.createSystem(topology, filename='mymodel.pt')
    """

    def __init__(self, name, model_path, atom_type_to_atomic_number):
        self.name = name
        self.model_path = model_path
        self.atom_type_to_atomic_number = atom_type_to_atomic_number


    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  filename: str = 'nequipmodel.pt',
                  implementation : str = None,
                  **args):
        

        
        import torch
        import openmmtorch
        from torch_nl import compute_neighborlist_n2



        # Create the PyTorch model that will be invoked by OpenMM.
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        atomic_numbers = [atom.element.atomic_number for atom in includedAtoms]


        class NequIPForce(torch.nn.Module):

            def __init__(self, model_path, atomic_numbers, atom_type_to_atomic_number):
                super(NequIPForce, self).__init__()

                # conversion constants
                self.register_buffer('nm_to_A',    torch.tensor(10.0))
                self.register_buffer('A_to_nm',    torch.tensor(0.1))
                self.register_buffer('kcal_to_kJ', torch.tensor(4.184))

                # TODO: using "nequip.scripts.deploy.load_deployed_model()" method gives torch errors
                # self.model, metadata = nequip.scripts.deploy.load_deployed_model("deplot.pth")

                # instead load model directly using torch.jit.load and get the metadata we need
                metadata = {k: "" for k in ["r_max","n_species","type_names"]}
                self.model = torch.jit.load(model_path, _extra_files=metadata)
                self.model.eval()

                # decode metadata
                metadata = {k: v.decode("ascii") for k, v in metadata.items()}

                self.r_max = torch.tensor(float(metadata["r_max"]))
                
                # check the type names in the model metadata match the type names given to the NNP from OpenMM
                type_names = str(metadata["type_names"]).split(" ")
                #print(type_names)
                assert(type_names == list(atom_type_to_atomic_number.keys()))
                type_name_to_type_index={ type_name : i for i,type_name in enumerate(type_names)}
            
                self.atomic_number_to_type_index = { atom_type_to_atomic_number[type_name] : type_name_to_type_index[type_name] for type_name in type_names }
                #print(self.atomic_number_to_type_index)

                self.atomic_numbers = torch.tensor(atomic_numbers,dtype=torch.long)
                self.N = len(atomic_numbers)
                self.atom_types = torch.tensor([self.atomic_number_to_type_index[x] for x in atomic_numbers],dtype=torch.long)


            def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):
                # setup positions
                positions = positions.to(dtype=torch.float32) * self.nm_to_A

                # create the input dict
                input_dict={}
                batch = torch.zeros(self.N,dtype=torch.long)
                input_dict["cell"]=torch.ones((3,3))
                self_interaction=False
                input_dict["pbc"]=torch.tensor([False, False, False])
                input_dict["atomic_numbers"] = self.atomic_numbers
                input_dict["atom_types"] = self.atom_types
                input_dict["pos"] = positions

                # compute edges
                # TODO: PBCs
                mapping, _ , shiftx_idx = compute_neighborlist_n2(cutoff=self.r_max, 
                                                                            pos=input_dict["pos"], 
                                                                            cell=input_dict["cell"], 
                                                                            pbc=input_dict["pbc"], 
                                                                            batch=batch, 
                                                                            self_interaction=self_interaction)
                
                edge_index = torch.stack((mapping[0], mapping[1]))
                
                input_dict["edge_index"] = edge_index
                input_dict["edge_cell_shift"] = shiftx_idx

                # predict
                out = self.model(input_dict)

                # return energy and forces
                energy = out["total_energy"]*self.kcal_to_kJ
                forces = out["forces"]*self.kcal_to_kJ/self.A_to_nm

                return (energy, forces)

        # TODO: is_periodic...
        #is_periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()
        #aniForce = ANIForce(model, species, atoms, is_periodic)
        

        nequipforce = NequIPForce(self.model_path, atomic_numbers, self.atom_type_to_atomic_number)
        
        
        # Convert it to TorchScript and save it.
        module = torch.jit.script(nequipforce)
        module.save(filename)

        # Create the TorchForce and add it to the System.

        force = openmmtorch.TorchForce(filename)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(False)
        force.setOutputsForces(True)
        system.addForce(force)

MLPotential.registerImplFactory('nequip', NequIPPotentialImplFactory())
