"""
macepotential.py: Implements the MACE potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021-2026 Stanford University and the Authors.
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
import openmm
from openmm import unit
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory
from openmmml.embeddings import utilities
from collections.abc import Iterable
from functools import partial
import numpy as np
import torch
import typing


class MACEPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates MACEPotentialImpl objects."""

    def createImpl(
        self, name: str, modelPath: str | None = None, **args
    ) -> MLPotentialImpl:
        return MACEPotentialImpl(name, modelPath)


class MACEPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the MACE potential.

    The MACE potential is constructed using MACE to build a PyTorch model,
    and then integrated into the OpenMM System using a TorchForce.
    This implementation supports both foundation models and locally trained MACE models.

    To use one of the pre-trained MACE foundation models, specify the model name. For example:

    >>> potential = MLPotential('mace-off23-small')

    Other available models include 'mace-off23-medium', 'mace-off23-large', 'mace-off24-medium',
    'mace-mpa-0-medium', 'mace-omat-0-small', 'mace-omat-0-medium', 'mace-omol-0-extra-large',
    'mace-les-off-small', 'mace-polar-1-small', 'mace-polar-1-medium', and 'mace-polar-1-large'.

    To use a locally trained MACE model, provide the path to the model file. For example:

    >>> potential = MLPotential('mace', modelPath='MACE.model')

    During system creation, you can optionally specify the precision of the model using the
    ``precision`` keyword argument. Supported options are 'single' and 'double'. For example:

    >>> system = potential.createSystem(topology, precision='single')

    By default, the implementation uses the precision of the loaded MACE model.
    According to the MACE documentation, 'single' precision is recommended for MD (faster but
    less accurate), while 'double' precision is recommended for geometry optimization.

    Additionally, you can request computation of the full atomic energy, including the atom
    self-energy, instead of the default interaction energy, by setting ``returnEnergyType`` to
    'energy'. For example:
    
    >>> system = potential.createSystem(topology, returnEnergyType='energy')

    The default is to compute the interaction energy, which can be made explicit by setting
    ``returnEnergyType='interaction_energy'``.

    Attributes
    ----------
    name : str
        The name of the MACE model.
    modelPath : str
        The path to the locally trained MACE model if ``name`` is 'mace'.
    """

    # (Function name, model name, restrictive license name or None, long-range, supports electrostatic embedding)
    KNOWN_MODELS = {
        'mace-off23-small': ('mace_off', 'small', 'ASL', False, False),
        'mace-off23-medium': ('mace_off', 'medium', 'ASL', False, False),
        'mace-off23-large': ('mace_off', 'large', 'ASL', False, False),
        'mace-off24-medium': ('mace_off', 'https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true', 'ASL', False, False),
        'mace-mpa-0-medium': ('mace_mp', 'medium-mpa-0', None, False, False),
        'mace-omat-0-small': ('mace_mp', 'small-omat-0', 'ASL', False, False),
        'mace-omat-0-medium': ('mace_mp', 'medium-omat-0', 'ASL', False, False),
        'mace-omol-0-extra-large': ('mace_omol', 'extra_large', 'ASL', False, False),
        'mace-les-off-small': ('mace_off', 'https://github.com/ChengUCB/les_fit/blob/main/MACELES-OFF/MACELES-OFF_small_converted.model?raw=true', 'CC BY-NC 4.0', True, False),
        'mace-polar-1-small': ('mace_polar', 'polar-1-s', None, True, True),
        'mace-polar-1-medium': ('mace_polar', 'polar-1-m', None, True, True),
        'mace-polar-1-large': ('mace_polar', 'polar-1-l', None, True, True),
    }

    def __init__(self, name: str, modelPath) -> None:
        """
        Initialize the MACEPotentialImpl.

        Parameters
        ----------
        name : str
            The name of the MACE model.
            Options include 'mace-off23-small', 'mace-off23-medium', 'mace-off23-large',
            'mace-off24-medium', 'mace-mpa-0-medium', 'mace-omat-0-small', 'mace-omat-0-medium',
            'mace-omol-0-extra-large', 'mace-les-off-small', 'mace-polar-1-small',
            'mace-polar-1-medium', 'mace-polar-1-large', and 'mace'.
        modelPath : str, optional
            The path to the locally trained MACE model if ``name`` is 'mace'.
        """
        self.name = name
        self.modelPath = modelPath

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Iterable[int] | None,
        forceGroup: int,
        precision: str | None = None,
        returnEnergyType: str | None = None,
        embedding: str = "mechanical",
        **args,
    ) -> None:
        """
        Add the MACEForce to the OpenMM System.

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
            If ``None``, the default precision of the model is used.
        returnEnergyType : str, optional
            Whether to return the interaction energy or the energy including the
            self-energy.  For mechanical embedding, supported options are
            'interaction_energy' (the default) and 'energy'.  For electrostatic
            embedding, 'energy' is the default and only supported option.
        embedding : str, optional
            The model-specific embedding method to use.  This will be set by
            `createMixedSystem()` as needed and should not be used directly.
        """
        try:
            from mace.tools import utils, to_one_hot, atomic_numbers_to_indices
            from mace.calculators.foundations_models import mace_off, mace_mp, mace_omol, mace_polar
        except ImportError as e:
            raise ImportError(f"Failed to import mace with error: {e}. Install mace with 'pip install mace-torch'.")
        try:
            from e3nn.util import jit
        except ImportError as e:
            raise ImportError(f"Failed to import e3nn with error: {e}. Install e3nn with 'pip install e3nn'.")

        # Load the model.

        device = self._getTorchDevice(args)
        if self.name in MACEPotentialImpl.KNOWN_MODELS:
            functions = {
                'mace_off': mace_off,
                'mace_mp': mace_mp,
                'mace_omol': mace_omol,
                'mace_polar': mace_polar,
            }
            fnName, name, restrictiveLicense, _, _ = MACEPotentialImpl.KNOWN_MODELS[self.name]
            model = functions[fnName](model=name, device=device, return_raw_model=True).to(device)
            if restrictiveLicense is not None:
                import logging
                logging.warning(f'The model {self.name} is distributed under the restrictive {restrictiveLicense} license.  Commercial use is not permitted.')
        elif self.name == "mace":
            if self.modelPath is not None:
                model = torch.load(self.modelPath, map_location=device).to(device)
            else:
                raise ValueError("No modelPath provided for local MACE model.")
        else:
            raise ValueError(f"Unsupported MACE model: {self.name}")

        # See what kind of embedding to do.

        if embedding == "mechanical":
            if returnEnergyType is None:
                returnEnergyType = "interaction_energy"
            elif returnEnergyType not in ("interaction_energy", "energy"):
                raise ValueError(f"Unsupported returnEnergyType {returnEnergyType!r} for mechanical embedding")

            doElectrostaticEmbedding = False

        elif embedding == "electrostatic":
            if returnEnergyType is None:
                returnEnergyType = "energy"
            elif returnEnergyType != "energy":
                raise ValueError(f"Unsupported returnEnergyType {returnEnergyType!r} for electrostatic embedding")

            doElectrostaticEmbedding = True

        else:
            raise RuntimeError("Invalid embedding method passed to MACEPotentialImpl.addForces")

        # Get the atomic numbers of the ML region.

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        atomicNumbers = [atom.element.atomic_number for atom in includedAtoms]

        # Set the precision that the model will be used with.

        modelDefaultDtype = next(model.parameters()).dtype
        if precision is None:
            dtype = modelDefaultDtype
        elif precision == "single":
            dtype = torch.float32
        elif precision == "double":
            dtype = torch.float64
        else:
            raise ValueError(f"Unsupported precision {precision} for the model. Supported values are 'single' and 'double'.")
        if dtype != modelDefaultDtype:
            print(f"Model dtype is {modelDefaultDtype} and requested dtype is {dtype}. The model will be converted to the requested dtype.")
            model = model.to(dtype)

        # One hot encoding of atomic numbers

        zTable = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        nodeAttrs = to_one_hot(
            torch.tensor(atomic_numbers_to_indices(atomicNumbers, z_table=zTable), dtype=torch.long, device=device).unsqueeze(-1),
            num_classes=len(zTable))

        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()

        # Create the PythonForce and add it to the System.

        if doElectrostaticEmbedding:
            model = ElectrostaticEmbeddingMACE(model, torch.tensor(args["mmCharges"], dtype=dtype, device=device))
        compute = partial(_computeMACE,
                          model=model,
                          ptr=torch.tensor([0, nodeAttrs.shape[0]], dtype=torch.long, device=device, requires_grad=False),
                          node_attrs=nodeAttrs.to(dtype),
                          batch=torch.zeros(nodeAttrs.shape[0], dtype=torch.long, device=device, requires_grad=False),
                          pbc=torch.tensor([periodic, periodic, periodic], dtype=torch.bool, device=device, requires_grad=False),
                          returnEnergyType=returnEnergyType,
                          charge=torch.tensor([float(args.get('charge', 0))], dtype=dtype, device=device, requires_grad=False),
                          multiplicity=torch.tensor([float(args.get('multiplicity', 1))], dtype=dtype, device=device, requires_grad=False),
                          periodic=periodic,
                          doElectrostaticEmbedding=doElectrostaticEmbedding,
                          mlIndices=args["mlIndices"] if doElectrostaticEmbedding else None,
                          mmIndices=args["mmIndices"] if doElectrostaticEmbedding else None)
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        if atoms is not None and not doElectrostaticEmbedding:
            force.setParticles(atoms)
        system.addForce(force)

    def getSupportedEmbeddings(self) -> list[str]:
        if self.name in MACEPotentialImpl.KNOWN_MODELS:
            _, _, _, _, supportsElectrostatic = MACEPotentialImpl.KNOWN_MODELS[self.name]
        else:
            # A custom model might support electrostatic embedding; an error
            # will be raised later if a particular custom model does not.
            supportsElectrostatic = True
        return ["electrostatic"] if supportsElectrostatic else []

    def getMLLongRange(self) -> bool | None:
        if self.name in MACEPotentialImpl.KNOWN_MODELS:
            _, _, _, longRange, _ = MACEPotentialImpl.KNOWN_MODELS[self.name]
            return longRange
        return None

    def createMixedSystem(self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: list[int],
        forceGroup: int,
        interpolate: bool,
        embedding: str,
        **args,
    ) -> openmm.System | dict[str, typing.Any]:

        if embedding != "electrostatic":
            raise RuntimeError("Invalid embedding method passed to MACEPotentialImpl.createMixedSystem")

        # Create the new system with ML-ML interactions to be computed by the ML
        # potential removed.

        periodic = system.usesPeriodicBoundaryConditions()
        newSystem = utilities.removeBonds(system, topology, atoms, True)
        numAtoms = newSystem.getNumParticles()

        allCharges = [0.0] * numAtoms
        for force in newSystem.getForces():
            if isinstance(force, openmm.NonbondedForce):
                # Get charges on all particles.

                for atom in range(numAtoms):
                    charge, _, _ = force.getParticleParameters(atom)
                    allCharges[atom] += charge.value_in_unit(unit.elementary_charge)

                # All of the LJ interactions in the ML region should be zeroed.
                # The ML-ML and ML-MM electrostatics should both be zeroed.

                for atom in atoms:
                    _, sigma, epsilon = force.getParticleParameters(atom)
                    force.setParticleParameters(atom, 0, sigma, epsilon)

                for iAtom1 in range(len(atoms)):
                    for iAtom2 in range(iAtom1):
                        force.addException(atoms[iAtom1], atoms[iAtom2], 0, 1, 0, True)

                # This may cause exceptions in the MM region to use PBCs, but
                # this should not ordinarily have any significant effects.

                force.setExceptionsUsePeriodicBoundaryConditions(periodic)

            elif isinstance(force, openmm.CustomNonbondedForce):
                utilities.makeCustomNonbondedExclusions(force, atoms)

        # Prepare inputs to MACE-POLAR.

        mlAtomSet = set(atoms)
        mmAtomList = sorted(set(range(numAtoms)) - mlAtomSet)
        embeddingArgs = {
            "embedding": embedding,
            "mmCharges": np.array([allCharges[atom] for atom in mmAtomList]),
            "mmIndices": np.array(mmAtomList, dtype=int),
            "mlIndices": np.array(atoms, dtype=int),
        }

        if interpolate:
            interpolator = utilities.InterpolationHelper()
            interpolator.addMLPotentialTerms(self, topology, atoms, forceGroup, **embeddingArgs, **args)
            interpolator.addMMBondedTerms(system, topology, atoms)
            interpolator.setupNonbonded(newSystem, system)
            interpolator.setupInterpolation(newSystem)

        else:
            self.addForces(topology, newSystem, atoms, forceGroup, **embeddingArgs, **args)

        return newSystem


def _computeMACE(state, model, ptr, node_attrs, batch, pbc, returnEnergyType, charge, multiplicity, periodic, doElectrostaticEmbedding, mlIndices, mmIndices):
    from mace.data.neighborhood import get_neighborhood
    energyScale = 96.4853
    lengthScale = 10.0
    allPositions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    if doElectrostaticEmbedding:
        positions = allPositions[mlIndices]
    else:
        positions = allPositions
    if periodic:
        cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    else:
        cell = np.identity(3, dtype=np.float64)
    dtype = node_attrs.dtype
    cutoff = float(model.r_max.detach())
    edgeIndex, shifts, _, _ = get_neighborhood(positions, cutoff, [periodic, periodic, periodic], cell)
    cellTensor = torch.tensor(cell, dtype=dtype, device=ptr.device)
    inputDict = {
        "ptr": ptr,
        "node_attrs": node_attrs,
        "batch": batch,
        "pbc": pbc,
        "positions": torch.tensor(positions, dtype=dtype, device=ptr.device),
        "edge_index": torch.tensor(edgeIndex, dtype=torch.int64, device=ptr.device),
        "shifts": torch.tensor(shifts, dtype=dtype, device=ptr.device),
        "cell": cellTensor,
        "rcell": 2 * torch.pi * torch.linalg.inv(cellTensor.mT),
        "volume": torch.linalg.det(cellTensor).reshape(1),
        "total_charge": charge,
        "total_spin": multiplicity,
        "external_field": torch.zeros((1, 3), dtype=dtype, device=ptr.device),
        "fermi_level": torch.zeros((1,), dtype=dtype, device=ptr.device)
    }
    if doElectrostaticEmbedding:
        inputDict["mm_positions"] = torch.tensor(allPositions[mmIndices], dtype=dtype, device=ptr.device, requires_grad=True)
    results = model(inputDict, compute_force=not doElectrostaticEmbedding)
    energy = float(results[returnEnergyType].detach())*energyScale
    forces = (results["forces"]*energyScale*lengthScale).detach().cpu().numpy()
    if doElectrostaticEmbedding:
        mmForces = (results["mm_forces"]*energyScale*lengthScale).detach().cpu().numpy()
        allForces = np.zeros_like(allPositions)
        allForces[mlIndices] = forces
        allForces[mmIndices] = mmForces
        forces = allForces
    print("This is Evan's code")
    return energy, forces


class ElectrostaticEmbeddingMACE(torch.nn.Module):
    def __init__(self, model, mmCharges: torch.Tensor):
        super().__init__()

        try:
            from graph_longrange.external_source_features import GTOElectrostaticExternalSourceFeatures
            from graph_longrange.external_source_energy import GTOElectrostaticExternalSourceEnergy
        except ImportError as e:
            raise ImportError(f"Failed to import graph_longrange with error: {e}. Install graph_longrange with 'pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.4'.")

        # Wrap parts of the model to prepare to receive external features.

        if not (hasattr(model, "electric_potential_descriptor") and hasattr(model, "coulomb_energy")):
            raise ValueError("The selected MACE model does not support electrostatic embedding")
        model.electric_potential_descriptor = GTOElectrostaticExternalSourceFeatures.from_features(model.electric_potential_descriptor, external_scale=0.5)
        model.coulomb_energy = GTOElectrostaticExternalSourceEnergy.from_energy(model.coulomb_energy)

        # Preprocess MM charges.

        mmChargeFeatures = torch.zeros((mmCharges.shape[0], (model.atomic_multipoles_max_l + 1) ** 2), dtype=mmCharges.dtype, device=mmCharges.device)
        mmChargeFeatures[:, 0] = mmCharges
        chargeTransform = getattr(model, "_charges_to_mul_ir", None)
        if chargeTransform is not None:
            mmChargeFeatures = chargeTransform(mmChargeFeatures)

        self.model = model
        self.mmChargeFeatures = mmChargeFeatures
        self.mmBatch = torch.zeros(mmCharges.shape[0], dtype=torch.long, device=mmCharges.device)
        self.r_max = model.r_max

    def forward(self, data: dict[str, torch.Tensor], *args, **kwargs):
        mmPositions = data["mm_positions"]
        mlPositions = data["positions"]

        self.model.electric_potential_descriptor.set_external_sources(external_feats=self.mmChargeFeatures, external_positions=mmPositions, external_batch=self.mmBatch)
        self.model.coulomb_energy.set_external_sources(external_feats=self.mmChargeFeatures, external_positions=mmPositions, external_batch=self.mmBatch)

        result = self.model(data, *args, **kwargs)
        energy = result["energy"]
        mlGradient, mmGradient = torch.autograd.grad(outputs=[energy], inputs=[mlPositions, mmPositions], grad_outputs=[torch.ones_like(energy)])
        result["forces"] = -mlGradient
        result["mm_forces"] = -mmGradient
        return result
