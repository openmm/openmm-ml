"""
nequippotential.py: Implements the NequIP potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2024 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Stephen Farr, Joao Morado

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

from typing import Iterable, List, Optional, Tuple

import openmm

from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory


class NequIPPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates NequipPotentialImpl objects."""

    def createImpl(
        self,
        name: str,
        modelPath: str,
        lengthScale: float,
        energyScale: float,
    ) -> MLPotentialImpl:
        return NequIPPotentialImpl(name, modelPath, lengthScale, energyScale)


class NequIPPotentialImpl(MLPotentialImpl):
    """
    This is the MLPotentialImpl implementing support for E(3)-equivariant
    interatomic potentials generated by NequIP or Allegro.

    The potential must be constructed using the NequIP/Allegro code to build and
    deploy a PyTorch model, which can then be integrated into OpenMM using a
    TorchForce. The TorchForce is used to add it to the OpenMM System. Note that
    you must provide a deployed model, as no general purpose model is available.

    To use a deployed model in openmm-ml, you must provide the path to the model,
    and the conversion factors between the model length and energy units and OpenMM
    units (nm and kJ/mol, respectively). For example:

    >>> potential = MLPotential('nequip',
                                modelPath='example_model_deployed.pth',
                                lengthScale=0.1, # Angstrom to nm
                                energyScale=4.184 # kcal/mol to kJ/mol
                                )

    During system creation, if the model was trained with custom atom types,
    you can specify this by passing to the ``atomTypes`` parameter a list of
    of integers corresponding to the nequip atom type for each particle that
    will be modeled using this potential. This argument should, therefore,
    have the same length as the number of ML atoms in the system.  Note that
    by default the model uses the atomic number to map the atom type. This
    will work if you trained your model using the standard
    ``chemical_symbols`` option.

    Additionally, you can specify the precision of
    the model using the ``precision`` keyword argument. Supported options are
    'single' and 'double'. For example:

    >>> system = potential.createSystem(topology, precision='single')

    By default, the implementation uses the precision of the loaded model.
    Note that models deployed before NequIP v0.6.0 don't contain information
    about their precision, so ``precision='double'`` should only be used if
    the model was explicitly trained with ``default_dtype=float64``, as by
    default the model is trained with ``default_dtype=float32``.
    """
    def __init__(
        self,
        name: str,
        modelPath: str,
        lengthScale: float,
        energyScale: float,
    ) -> None:
        """
        Initialize the NequIPPotentialImpl.

        Parameters
        ----------
        name : str
            The name specified by the MLPotential constructor, viz. 'nequip'.
        modelPath : str, optional
            The path to the deployed model.
        lengthScale : float
            The energy conversion factor from the model units to kJ/mol.
        energyScale : float
            The length conversion factor from the model units to nanometers.
        """
        self.name = name
        self.modelPath = modelPath
        self.lengthScale = lengthScale
        self.energyScale = energyScale

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        precision: Optional[str] = None,
        atomTypes: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Add the NequIPForce to the OpenMM System.

        Parameters
        ----------
        topology : openmm.app.Topology
            The topology of the system.
        system : openmm.System
            The system to which the force will be added.
        atoms : iterable of int
            The indices of the atoms to include in the model. If ``None``, all atoms are included.
        forceGroup : int
            The force group to which the force should be assigned.
        precision : str, optional
            The precision of the model. Supported options are 'single' and 'double'.
            If ``None``, the default precision of the model is used. This is the
            recommended option. Models deployed before NequIP v0.6.0 don't contain
            information about their precision, so ``precision='double'`` should only be
            used if the model was explicitly trained with ``default_dtype=float64``, 
            as by default the model is trained with ``default_dtype=float32``.
        atomTypes : List[int], optional
            A list of integers corresponding to the nequip atom type for each ML atom in the system.
            This is only required if the model was trained with custom atom types. If ``None``,
            the atomic number is used to determine the atom type. This list should have the same
            length as the number of ML atoms in the system.
        """
        import openmmtorch
        import torch

        try:
            from NNPOps.neighbors import getNeighborPairs
        except ImportError as e:
            raise ImportError(
                f"Failed to import NNPOps with error: {e}. "
                "Install NNPOps with 'conda install -c conda-forge nnpops'."
            )

        try:
            import nequip.scripts.deploy
        except ImportError as e:
            raise ImportError(
                f"Failed to import NequIP with error: {e}. "
                "Install NequIP with 'pip install git+https://github.com/mir-group/nequip@develop'."
            )
        
        # Load the model to the CPU.
        model, metadata = nequip.scripts.deploy.load_deployed_model(
            self.modelPath, device="cpu", freeze=False
        )

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]

        # Try to infer the model dtype from the metadata.
        # If not present, default to float32.
        modelDefaultDtype = {"float32": torch.float32, "float64": torch.float64}[
            metadata.get("model_dtype", "float32")
        ]
        # Set the precision that the model will be used with.
        if precision is None:
            dtype = modelDefaultDtype
        elif precision == "single":
            dtype = torch.float32
        elif precision == "double":
            dtype = torch.float64
        else:
            raise ValueError(
                f"Unsupported precision {precision} for the model. "
                "Supported values are 'single' and 'double'."
            )

        if dtype != modelDefaultDtype:
            print(
                f"Model dtype in metadata is {modelDefaultDtype} "
                f"and requested dtype is {dtype}. "
                "The model will be converted to the requested dtype. "
                "Make sure this is the precision the model was trained with."
            )

        # Get the atom types
        if atomTypes is None:
            typeNames = metadata[nequip.scripts.deploy.TYPE_NAMES_KEY].split(" ")
            typeNameToTypeIndex = {
                typeNames: i for i, typeNames in enumerate(typeNames)
            }
            atomTypes = [
                typeNameToTypeIndex[atom.element.symbol] for atom in includedAtoms
            ]
        else:
            if len(atomTypes) != len(includedAtoms):
                raise ValueError(
                    "The length of atomTypes must be equal to the number of ML atoms in the system."
                )

        # Get the r_max from the metadata.
        r_max = float(metadata[nequip.scripts.deploy.R_MAX_KEY])

        class NequIPForce(torch.nn.Module):
            """
            NequIPForce is a PyTorch module that wraps a NequIP model.

            Attributes
            ----------
            dtype : torch.dtype
                The precision of the model.
            model : str
                The loaded deployed NequIP model.
            energyScale : float
                The length conversion factor from the model units to nanometers.
            lengthScale : float
                The energy conversion factor from the model units to kJ/mol.
            r_max : torch.Tensor
                The maximum distance for the neighbor search.
            indices : torch.Tensor or None
                The indices of the atoms to model using this potential.
                If ``None``, all atoms are considered as ML atoms.
            """

            def __init__(
                self,
                model: torch.jit._script.RecursiveScriptModule,
                atoms: Optional[Iterable[int]],
                periodic: bool,
                lengthScale: float,
                energyScale: float,
                atomTypes: List[str],
                r_max: float,
                dtype: torch.dtype,
            ) -> None:
                """
                Initialize the NequIPForce.

                Parameters
                ----------
                model : torch.jit._script.RecursiveScriptModule
                    The loaded deployed NequIP model.
                atoms : iterable of int
                    Indices of the atoms to use with the model. If ``None``, all atoms are used.
                periodic : bool
                    Whether the system is periodic.
                lengthScale : float
                    The energy conversion factor from the model units to kJ/mol.
                energyScale : float
                    The length conversion factor from the model units to nanometers.
                atomTypes : List[int]
                    The nequip atom types for the model.
                r_max : float
                    The maximum distance for the neighbor search.
                dtype : torch.dtype
                    The precision of the model.
                """
                super(NequIPForce, self).__init__()

                self.dtype = dtype
                self.model = model.to(self.dtype)
                self.energyScale = energyScale
                self.lengthScale = lengthScale
                self.r_max = torch.tensor(r_max, dtype=self.dtype, requires_grad=False)

                if atoms is None:
                    self.indices = None
                else:
                    self.indices = torch.tensor(sorted(atoms), dtype=torch.int64)

                # Create the default input dict
                self.register_buffer("atom_types", torch.tensor(atomTypes, dtype=torch.long, requires_grad=False))
                self.register_buffer("pbc", torch.tensor([periodic, periodic, periodic], dtype=torch.bool, requires_grad=False))

            def _getNeighborPairs(
                self, positions: torch.Tensor, cell: Optional[torch.Tensor]
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                """
                Get the shift and edge indices.

                Notes
                -----
                This method calculates the shifts and edge indices by determining neighbor pairs (``neighbors``)
                and respective wrapped distances (``wrappedDeltas``) using ``NNPOps.neighbors.getNeighborPairs``.
                After obtaining the ``neighbors`` and ``wrappedDeltas``, the pairs with negative indices (r>cutoff)
                are filtered out, and the edge indices and shift indices are finally calculated.

                Parameters
                ----------
                positions : torch.Tensor
                    The positions of the atoms.
                cell : torch.Tensor
                    The cell vectors.

                Returns
                -------
                edgeIdx : torch.Tensor
                    The edge indices tensor giving center -> neighbor relations.
                shiftIdx : torch.Tensor
                    The shift indices tensor indicating how many periodic cells each edge crosses in each cell vector. 
                """
                # Get the neighbor pairs, shifts and edge indices.
                neighbors, wrappedDeltas, _, _ = getNeighborPairs(
                    positions, self.r_max, -1, cell
                )
                mask = neighbors >= 0
                neighbors = neighbors[mask].view(2, -1)
                wrappedDeltas = wrappedDeltas[mask[0], :]

                edgeIdx = torch.hstack((neighbors, neighbors.flip(0))).to(torch.int64)
                if cell is not None:
                    deltas = positions[edgeIdx[0]] - positions[edgeIdx[1]]
                    wrappedDeltas = torch.vstack((wrappedDeltas, -wrappedDeltas))
                    shiftIdx = torch.mm(deltas - wrappedDeltas, torch.linalg.inv(cell))
                else:
                    shiftIdx = torch.zeros(
                        (edgeIdx.shape[1], 3),
                        dtype=self.dtype,
                        device=positions.device,
                    )

                return edgeIdx, shiftIdx

            def forward(
                self, positions: torch.Tensor, boxvectors: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                """
                Forward pass of the NequIP model.

                Parameters
                ----------
                positions : torch.Tensor
                    The positions of the atoms.
                boxvectors : torch.Tensor, optional
                    The box vectors.

                Returns
                -------
                energy, forces : Tuple[torch.Tensor, torch.Tensor]
                    The predicted energy in kJ/mol and forces in kJ/(mol*nm).
                """         
                # Created a padded forces tensor.     
                forces_padded = torch.zeros_like(positions).to(self.dtype)

                # Setup positions and cell.
                if self.indices is not None:
                    positions = positions[self.indices]

                positions = positions / self.lengthScale

                if boxvectors is not None:
                    cell = boxvectors.to(self.dtype) / self.lengthScale
                else:
                    cell = None

                # Get the shifts and edge indices.
                edgeIdx, shiftIdx = self._getNeighborPairs(positions, cell)

                # Create the input dictionary.
                inputDict = {
                    "pos": positions,
                    "atom_types": self.atom_types,
                    "edge_index": edgeIdx,
                    "edge_cell_shift": shiftIdx,
                    "pbc": self.pbc,
                    "cell": cell if cell is not None else torch.zeros(3, 3, device=positions.device, dtype=self.dtype),
                }

                # Predict the energy and forces.
                out = self.model(inputDict)
                energy = out["total_energy"] * self.energyScale

                if self.indices is None:
                    forces_padded = out["forces"] * self.energyScale / self.lengthScale
                else:
                    forces_padded[self.indices, :] = out["forces"] * self.energyScale / self.lengthScale
            
                return (energy, forces_padded)

        isPeriodic = (
            topology.getPeriodicBoxVectors() is not None
        ) or system.usesPeriodicBoundaryConditions()

        nequipForce = NequIPForce(
            model,
            atoms,
            isPeriodic,
            self.lengthScale,
            self.energyScale,
            atomTypes,
            r_max,
            dtype,
        )

        # Convert it to TorchScript
        module = torch.jit.script(nequipForce)

        # Create the TorchForce and add it to the System.
        force = openmmtorch.TorchForce(module)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(isPeriodic)
        force.setOutputsForces(True)
        system.addForce(force)