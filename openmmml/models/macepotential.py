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
from typing import Iterable, Optional, Union

class MACEPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates MACEPotentialImpl objects."""

    def createImpl(self, name: str, model_path: str, **args) -> MLPotentialImpl:
        return MACEPotentialImpl(name, model_path)

class MACEPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the MACE potential.

    The potential is implemented using MACE to build a PyTorch model.  A
    TorchForce is used to add it to the OpenMM System.  

    TorchForce requires the model to be saved to disk in a separate file.  By default
    it writes a file called 'macemodel.pt' in the current working directory.  You can
    use the filename argument to specify a different name.  For example,

    >>> system = potential.createSystem(topology, filename='mymodel.pt')
    """

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  filename: str = 'macemodel.pt',
                  implementation : str = None,
                  device: str = None,
                  dtype: str = "float64",
                  **args):
        

        import torch
        import openmmtorch
        from torch_nl import compute_neighborlist
        from e3nn.util import jit
        from mace.tools import utils, to_one_hot, atomic_numbers_to_indices
        


        # Create the PyTorch model that will be invoked by OpenMM.

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            # check if atoms needs to be ordered
            includedAtoms = [includedAtoms[i] for i in atoms]
        

        class MACEForce(torch.nn.Module):

            def __init__(self, model_path, atomic_numbers, indices, periodic, device, dtype=torch.float64):
                super(MACEForce, self).__init__()

                if device is None: # use cuda if available
                    self.device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

                else: # unless user has specified the device 
                    self.device=torch.device(device)

                self.default_dtype = dtype
                torch.set_default_dtype(self.default_dtype)

                print("Running MACEForce on device: ", self.device, " with dtype: ", self.default_dtype)
                

                # conversion constants 
                self.nm_to_distance = 10.0 # nm->A
                self.distance_to_nm = 0.1 # A->nm
                self.energy_to_kJ = 96.49 # eV->kJ

                self.model = torch.load(model_path,map_location=device)
                self.model.to(self.default_dtype)
                self.model.eval()

                #print(self.model)
                #for name, param in self.model.state_dict().items():
                #    print(name, param.size())
                
                self.r_max = self.model.r_max
                self.z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])

                self.model = jit.compile(self.model)
                
                # setup input
                N=len(atomic_numbers)
                self.ptr = torch.tensor([0,N],dtype=torch.long, device=self.device)
                self.batch = torch.zeros(N, dtype=torch.long, device=self.device)
                
                # one hot encoding of atomic number
                self.node_attrs =to_one_hot(
                        torch.tensor(atomic_numbers_to_indices(atomic_numbers, z_table=self.z_table), dtype=torch.long, device=self.device).unsqueeze(-1),
                        num_classes=len(self.z_table),
                    )

                if periodic:
                    self.pbc=torch.tensor([True, True, True], device=self.device)
                else:
                    self.pbc=torch.tensor([False, False, False], device=self.device)


                self.compute_nl = compute_neighborlist

                if indices is None:
                    self.indices = None
                else:
                    self.indices = torch.tensor(indices, dtype=torch.int64)

            

            def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):
                # setup positions
                positions = positions.to(dtype=self.default_dtype)
                if self.indices is not None:
                    positions = positions[self.indices]

                positions = positions*self.nm_to_distance

                if boxvectors is not None:
                    cell = boxvectors.to(dtype=self.default_dtype) * self.nm_to_distance
                else:
                    cell = torch.eye(3, device=self.device)

                # compute edges
                mapping, _ , shifts_idx = self.compute_nl(cutoff=self.r_max, 
                                                                            pos=positions, 
                                                                            cell=cell, 
                                                                            pbc=self.pbc, 
                                                                            batch=self.batch, 
                                                                            self_interaction=False)
                
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

                energy = out["energy"]
                if energy is None:
                    energy = torch.tensor(0.0, device=self.device)
                
                # return energy 
                energy = energy*self.energy_to_kJ

                return energy
            

        is_periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()


        atomic_numbers = [atom.element.atomic_number for atom in includedAtoms]

        torch_dtype = {"float32":torch.float32, "float64":torch.float64}[dtype]

        maceforce = MACEForce(self.model_path, atomic_numbers, atoms, is_periodic, device, dtype=torch_dtype)
        
        # Convert it to TorchScript and save it.
        module = torch.jit.script(maceforce)
        module.save(filename)

        # Create the TorchForce and add it to the System.
        force = openmmtorch.TorchForce(filename)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(is_periodic)
        #force.setOutputsForces(True)
        system.addForce(force)

MLPotential.registerImplFactory('mace', MACEPotentialImplFactory())
