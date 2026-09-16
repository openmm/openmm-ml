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
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
from openmmml.embeddings import utilities
from typing import Iterable, Optional
from functools import partial
import numpy as np


def _prepare_external_sources(model, data, compute_force: bool):
    import torch

    positions = data.get("mm_positions")
    charges = data.get("mm_charges")
    if positions is None or charges is None:
        return None

    ml_positions = data["positions"]
    positions = positions.to(ml_positions).clone().requires_grad_(compute_force)
    charges = charges.to(ml_positions).reshape(-1)
    if positions.shape[0] != charges.shape[0]:
        raise ValueError("MM positions and charges must have the same length.")

    width = (int(model.atomic_multipoles_max_l) + 1) ** 2
    features = torch.zeros((len(charges), width), device=ml_positions.device,
                           dtype=ml_positions.dtype)
    features[:, 0] = charges

    transform = getattr(model, "_charges_to_mul_ir", None)
    if transform is not None:
        features = transform(features)

    batch = data.get("mm_source_batch")
    if batch is None:
        if data["pbc"].reshape(-1, 3).shape[0] != 1:
            raise ValueError(
                "mm_source_batch is required for batched PolarMACE inputs."
            )
        batch = torch.zeros(len(positions), dtype=torch.long, device=positions.device)
    else:
        batch = batch.to(device=positions.device, dtype=torch.long).reshape(-1)
        if batch.shape[0] != positions.shape[0]:
            raise ValueError("mm_source_batch and mm_positions must have the same length.")
    return {"positions": positions, "features": features, "batch": batch}


def _enable_polarmace_external_sources(model):
    """Wrap PolarMACE with dynamic MM electrostatic sources in eager mode."""
    import torch

    if getattr(model, "supports_external_electrostatics", False):
        return model
    if model.__class__.__name__ != "PolarMACE":
        raise TypeError(
            "External electrostatic sources require a PolarMACE model; got "
            f"{model.__class__.__name__}."
        )

    try:
        from graph_longrange.external_source_energy import (
            GTOElectrostaticExternalSourceEnergy,
        )
        from graph_longrange.external_source_features import (
            GTOElectrostaticExternalSourceFeatures,
        )
    except ImportError as exc:
        raise ImportError(
            "PolarMACE electrostatic embedding requires the external-source energy "
            "and feature blocks from graph_longrange. Install the external_field "
            "branch:\n"
            "  pip install 'git+https://github.com/WillBaldwin0/"
            "graph_electrostatics.git@external_field'"
        ) from exc

    model.electric_potential_descriptor = (
        GTOElectrostaticExternalSourceFeatures.from_features(
            model.electric_potential_descriptor,
            # PolarMACE has two spin channels. Each receives half of the
            # physical external potential.
            external_scale=0.5,
        )
    )
    model.coulomb_energy = GTOElectrostaticExternalSourceEnergy.from_energy(
        model.coulomb_energy
    )

    class PolarMACEExternalSources(torch.nn.Module):
        supports_external_electrostatics = True

        def __init__(self, wrapped):
            super().__init__()
            self.model = wrapped

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.model, name)

        def forward(
            self,
            data,
            training: bool = False,
            compute_force: bool = True,
            compute_virials: bool = False,
            compute_stress: bool = False,
            compute_displacement: bool = False,
            compute_hessian: bool = False,
            compute_edge_forces: bool = False,
            compute_atomic_stresses: bool = False,
            **kwargs,
        ):
            external = _prepare_external_sources(self.model, data, compute_force)
            if external is None:
                return self.model(
                    data,
                    training=training,
                    compute_force=compute_force,
                    compute_virials=compute_virials,
                    compute_stress=compute_stress,
                    compute_displacement=compute_displacement,
                    compute_hessian=compute_hessian,
                    compute_edge_forces=compute_edge_forces,
                    compute_atomic_stresses=compute_atomic_stresses,
                    **kwargs,
                )
            if any(
                (
                    compute_virials,
                    compute_stress,
                    compute_displacement,
                    compute_hessian,
                    compute_edge_forces,
                    compute_atomic_stresses,
                )
            ):
                raise NotImplementedError(
                    "The OpenMM external-source adapter currently supports "
                    "energies and Cartesian forces only."
                )

            external_kwargs = {
                "external_feats": external["features"],
                "external_positions": external["positions"],
                "external_batch": external["batch"],
            }
            self.model.electric_potential_descriptor.set_external_sources(
                **external_kwargs
            )
            self.model.coulomb_energy.set_external_sources(**external_kwargs)
            try:
                result = self.model(
                    data,
                    training=training,
                    compute_force=False,
                    compute_virials=False,
                    compute_stress=False,
                    compute_displacement=False,
                    compute_hessian=False,
                    compute_edge_forces=False,
                    compute_atomic_stresses=False,
                    **kwargs,
                )
                if compute_force:
                    ml_gradient, mm_gradient = torch.autograd.grad(
                        outputs=[result["energy"]],
                        inputs=[data["positions"], external["positions"]],
                        grad_outputs=[torch.ones_like(result["energy"])],
                        create_graph=training,
                        retain_graph=training,
                        allow_unused=True,
                    )
                    result["forces"] = (
                        torch.zeros_like(data["positions"])
                        if ml_gradient is None
                        else -ml_gradient
                    )
                    result["mm_forces"] = (
                        torch.zeros_like(external["positions"])
                        if mm_gradient is None
                        else -mm_gradient
                    )
                else:
                    result["mm_forces"] = None
                return result
            finally:
                self.model.electric_potential_descriptor.clear_external_sources()
                self.model.coulomb_energy.clear_external_sources()

    return PolarMACEExternalSources(model)


class MACEPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates MACEPotentialImpl objects."""

    def createImpl(
        self, name: str, modelPath: Optional[str] = None, **args
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
    'mace-les-off-small', and the PolarMACE models 'mace-polar-1-small',
    'mace-polar-1-medium', and 'mace-polar-1-large'.  The PolarMACE models are
    the ones that support electrostatic embedding.

    To use a locally trained MACE model, provide the path to the model file. For example:

    >>> potential = MLPotential('mace', modelPath='MACE.model')

    During system creation, you can optionally specify the precision of the model using the
    ``precision`` keyword argument. Supported options are 'single' and 'double'. For example:

    >>> system = potential.createSystem(topology, precision='single')

    By default, the implementation uses the precision of the loaded MACE model.
    Single precision is faster; double precision is more accurate.

    By default the reported energy is ``interaction_energy``. PolarMACE is an
    exception: it automatically uses the full ``energy`` output because that is
    the scalar whose gradient w.r.t. positions is reported as the force.

    PolarMACE automatically uses ``energy`` so its electrostatic forces and
    reported energy remain consistent.

    Attributes
    ----------
    name : str
        The name of the MACE model.
    modelPath : str
        The path to the locally trained MACE model if ``name`` is 'mace'.
    """

    # (loader, model name, restrictive license, long-range, accepts MM charges)
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
        self._preloadedModel = None

    def _loadModel(self, args):
        """Load a MACE model and place it on the requested device."""
        import torch
        try:
            from mace.calculators.foundations_models import mace_off, mace_mp, mace_omol, mace_polar
        except ImportError as e:
            raise ImportError(f"Failed to import mace with error: {e}. Install mace with 'pip install mace-torch'.")

        device = self._getTorchDevice(args)
        preloaded, self._preloadedModel = self._preloadedModel, None
        if preloaded is not None and preloaded[1] == device:
            return preloaded[0], device

        if self.name in MACEPotentialImpl.KNOWN_MODELS:
            loaders = {
                'mace_off': mace_off,
                'mace_mp': mace_mp,
                'mace_omol': mace_omol,
                'mace_polar': mace_polar,
            }
            loader_name, model_name, restrictive_license, _, _ = self.KNOWN_MODELS[self.name]
            model = loaders[loader_name](model=model_name, device=device, return_raw_model=True).to(device)
            if restrictive_license is not None:
                import logging
                logging.warning(f'The model {self.name} is distributed under the restrictive {restrictive_license} license. Commercial use is not permitted.')
        elif self.name == "mace":
            if self.modelPath is None:
                raise ValueError("No modelPath provided for local MACE model.")
            model = torch.load(self.modelPath, map_location=device)
            if hasattr(model, "to"):
                model = model.to(device)
        else:
            raise ValueError(f"Unsupported MACE model: {self.name}")
        if model.__class__.__name__ == "PolarMACE":
            model = _enable_polarmace_external_sources(model)
        return model, device

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        precision: Optional[str] = None,
        returnEnergyType: str = "interaction_energy",
        embedding: str = "mechanical",
        customNonbondedChargeParameter: Optional[str] = None,
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
            Which scalar from the MACE model output is reported to OpenMM as
            the potential energy. The default is ``'interaction_energy'`` for
            ordinary MACE and ``'energy'`` for PolarMACE, whose force gradient
            includes additional electrostatic terms.
        embedding : {"mechanical", "electrostatic"}
            Which embedding method the caller is implementing. Set by
            ``createMixedSystem``; there is normally no reason to pass it here
            directly. ``mechanical`` (the default) does not pass MM positions
            or charges into MACE. ``electrostatic`` passes them into PolarMACE
            and scatters the returned ``mm_forces`` back onto the MM atoms; the
            removal of the classical ML-MM Coulomb that this assumes is done by
            ``createMixedSystem``, not here. It is an error to request it for a
            model that cannot accept MM charges.
        """
        import torch
        try:
            from mace.tools import utils, to_one_hot, atomic_numbers_to_indices
        except ImportError as e:
            raise ImportError(f"Failed to import mace with error: {e}. Install mace with 'pip install mace-torch'.")

        assert returnEnergyType in ["interaction_energy", "energy"], f"Unsupported returnEnergyType: '{returnEnergyType}'. Supported options are 'interaction_energy' or 'energy'."

        model, device = self._loadModel(args)
        if model.__class__.__name__ in ("PolarMACE", "PolarMACEExternalSources"):
            returnEnergyType = "energy"

        use_mm_embedding = _should_use_mm_embedding(model, atoms, embedding)

        included_atoms = list(topology.atoms())
        if atoms is not None:
            included_atoms = [included_atoms[i] for i in atoms]
        atomic_numbers = [atom.element.atomic_number for atom in included_atoms]

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

        model_device = device
        try:
            model_device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            pass

        zTable = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        nodeAttrs = to_one_hot(
            torch.tensor(atomic_numbers_to_indices(atomic_numbers, z_table=zTable), dtype=torch.long, device=model_device).unsqueeze(-1),
            num_classes=len(zTable))

        indices = None if atoms is None else np.array(atoms)
        mmInfo = None
        if use_mm_embedding:
            mmInfo = _prepareMMEmbedding(system, atoms, customNonbondedChargeParameter)
        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()

        compute = partial(_computeMACE,
                          model=model,
                          ptr=torch.tensor([0, nodeAttrs.shape[0]], dtype=torch.long, device=model_device, requires_grad=False),
                          node_attrs=nodeAttrs.to(dtype),
                          batch=torch.zeros(nodeAttrs.shape[0], dtype=torch.long, device=model_device, requires_grad=False),
                          pbc=torch.tensor([periodic, periodic, periodic], dtype=torch.bool, device=model_device, requires_grad=False),
                          returnEnergyType=returnEnergyType,
                          charge=torch.tensor([float(args.get('charge', 0))], dtype=dtype, device=model_device, requires_grad=False),
                          multiplicity=torch.tensor([float(args.get('multiplicity', 1))], dtype=dtype, device=model_device, requires_grad=False),
                          indices=indices,
                          periodic=periodic,
                          mmInfo=mmInfo)
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)

    def getMLLongRange(self) -> bool | None:
        if self.name in MACEPotentialImpl.KNOWN_MODELS:
            _, _, _, longRange, _ = MACEPotentialImpl.KNOWN_MODELS[self.name]
            return longRange
        return None

    def getSupportedEmbeddings(self) -> list[str]:
        if self.name in MACEPotentialImpl.KNOWN_MODELS:
            return ["electrostatic"] if self.KNOWN_MODELS[self.name][4] else []
        return ["electrostatic"]

    def createMixedSystem(self,
                          topology: openmm.app.Topology,
                          system: openmm.System,
                          atoms: list[int],
                          forceGroup: int,
                          interpolate: bool,
                          embedding: str,
                          customNonbondedHasCharges: Optional[bool] = None,
                          customNonbondedChargeParameter: Optional[str] = None,
                          **args) -> openmm.System:
        """Create a mixed system using electrostatic embedding.

        PolarMACE receives MM positions and charges and computes all ML/MM
        electrostatics. The conventional ML charges and ML-region bonded terms
        are removed; Lennard-Jones and MM/MM terms remain in the force field.
        Exactly one NonbondedForce is required. CustomNonbondedForce charge
        handling must be declared with ``customNonbondedHasCharges`` and, when
        needed, ``customNonbondedChargeParameter``.
        """

        if embedding != "electrostatic":
            raise ValueError(f"Unsupported embedding type: {embedding}")

        # Validate before modifying the input system.

        model, device = self._loadModel(args)
        if not _supports_mm_embedding(model):
            raise ValueError(
                f"embedding='{embedding}' requires a model that accepts MM charges "
                f"and positions (PolarMACE); got {model.__class__.__name__}."
            )

        if interpolate:
            raise ValueError("Electrostatic embedding does not support interpolation.")

        periodic = system.usesPeriodicBoundaryConditions()

        nonbondedForces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
        if len(nonbondedForces) > 1:
            # The MM charges handed to the model are read from a single
            # NonbondedForce, so several of them are ambiguous.
            raise ValueError("Multiple NonbondedForce objects encountered; electrostatic embedding requires exactly one.")

        for force in nonbondedForces:
            for index in range(force.getNumParticleParameterOffsets()):
                if force.getParticleParameterOffset(index)[2] != 0:
                    raise ValueError("Electrostatic embedding does not support charge parameter offsets.")
            for index in range(force.getNumExceptionParameterOffsets()):
                if force.getExceptionParameterOffset(index)[2] != 0:
                    raise ValueError("Electrostatic embedding does not support charge parameter offsets.")

        if any(isinstance(f, openmm.CustomNonbondedForce) for f in system.getForces()):
            # A CustomNonbondedForce's energy expression is arbitrary, so
            # whether it contains electrostatics cannot be determined here.
            if customNonbondedHasCharges is None:
                raise ValueError("The System contains a CustomNonbondedForce and it is unknown whether it includes electrostatic interactions; pass customNonbondedHasCharges to specify.")
            if customNonbondedHasCharges and customNonbondedChargeParameter is None:
                raise ValueError("A CustomNonbondedForce includes electrostatic interactions, so customNonbondedChargeParameter must name the per-particle parameter holding the charge.")

        newSystem = utilities.removeBonds(system, topology, atoms, True)
        atomSet = set(atoms)

        for force in newSystem.getForces():
            if isinstance(force, openmm.NonbondedForce):

                for atom in atoms:
                    charge, sigma, epsilon = force.getParticleParameters(atom)
                    force.setParticleParameters(atom, 0.0, sigma, epsilon)

                for index in range(force.getNumExceptions()):
                    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(index)
                    if p1 in atomSet or p2 in atomSet:
                        force.setExceptionParameters(index, p1, p2, 0.0, sigma, epsilon)

                for i in range(len(atoms)):
                    for j in range(i):
                        force.addException(atoms[i], atoms[j], 0, 1, 0, True)

                force.setExceptionsUsePeriodicBoundaryConditions(periodic)

            elif isinstance(force, openmm.CustomNonbondedForce):

                if customNonbondedChargeParameter is not None:
                    names = [force.getPerParticleParameterName(i)
                             for i in range(force.getNumPerParticleParameters())]
                    if customNonbondedChargeParameter not in names:
                        raise ValueError(f"A CustomNonbondedForce has no per-particle parameter '{customNonbondedChargeParameter}'; it defines {names}.")
                    chargeIndex = names.index(customNonbondedChargeParameter)
                    for atom in atoms:
                        parameters = list(force.getParticleParameters(atom))
                        parameters[chargeIndex] = 0.0
                        force.setParticleParameters(atom, parameters)

                utilities.addCustomNonbondedExclusions(force, atoms)

        self._preloadedModel = (model, device)
        try:
            self.addForces(topology, newSystem, atoms, forceGroup, embedding=embedding,
                       customNonbondedChargeParameter=customNonbondedChargeParameter, **args)
        finally:
            self._preloadedModel = None

        return newSystem


def _supports_mm_embedding(model) -> bool:
    return bool(getattr(model, "supports_external_electrostatics", False))


def _should_use_mm_embedding(model, atoms: Optional[Iterable[int]], embedding: str) -> bool:
    """Validate the only MACE-specific embedding: electrostatic."""
    if embedding == "mechanical":
        return False
    if embedding != "electrostatic":
        raise ValueError(f"Unsupported embedding mode '{embedding}'.")
    if not _supports_mm_embedding(model):
        raise ValueError(
            f"embedding='{embedding}' requires a model that accepts MM charges "
            f"and positions (PolarMACE); got {model.__class__.__name__}."
        )
    if atoms is None:
        raise ValueError(
            f"embedding='{embedding}' requires an ML subset; it cannot be used "
            "with createSystem()."
        )
    return True


def _prepareMMEmbedding(system: openmm.System, atoms: Optional[Iterable[int]],
                        customNonbondedChargeParameter: Optional[str] = None):
    """Extract the MM complement and its charges from the system's NonbondedForce."""
    if atoms is None:
        return None

    num_particles = int(system.getNumParticles())
    ml_atoms = np.asarray(list(atoms), dtype=np.int64)
    ml_set = set(int(i) for i in ml_atoms.tolist())
    mm_atoms = np.asarray(
        [i for i in range(num_particles) if i not in ml_set], dtype=np.int64
    )

    # The charges given to the model have to come from wherever the force field
    # actually keeps them, which is the same force createMixedSystem zeroed the
    # ML charges in.  When that is a CustomNonbondedForce the caller has named
    # the parameter holding them.
    mm_charges = np.empty(len(mm_atoms), dtype=np.float64)

    if customNonbondedChargeParameter is not None:
        custom = None
        for force in system.getForces():
            if isinstance(force, openmm.CustomNonbondedForce):
                names = [force.getPerParticleParameterName(i)
                         for i in range(force.getNumPerParticleParameters())]
                if customNonbondedChargeParameter in names:
                    custom = force
                    chargeIndex = names.index(customNonbondedChargeParameter)
                    break
        if custom is None:
            raise ValueError(f"No CustomNonbondedForce defines a per-particle parameter '{customNonbondedChargeParameter}'.")
        for row, atom_index in enumerate(mm_atoms):
            mm_charges[row] = custom.getParticleParameters(int(atom_index))[chargeIndex]
        return {
            "ml_atoms": ml_atoms,
            "mm_atoms": mm_atoms,
            "mm_charges": mm_charges,
        }

    nonbonded = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded = force
            break
    if nonbonded is None:
        raise ValueError(
            "PolarMACE MM embedding requires a NonbondedForce to source MM charges."
        )

    for row, atom_index in enumerate(mm_atoms):
        charge, _, _ = nonbonded.getParticleParameters(int(atom_index))
        mm_charges[row] = charge.value_in_unit(unit.elementary_charge)

    return {
        "ml_atoms": ml_atoms,
        "mm_atoms": mm_atoms,
        "mm_charges": mm_charges,
    }


def _computeMACE(state, model, ptr, node_attrs, batch, pbc, returnEnergyType, charge, multiplicity, indices, periodic, mmInfo=None):
    import torch
    from mace.data.neighborhood import get_neighborhood
    energyScale = 96.4853
    lengthScale = 10.0
    positions_full = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    numAtoms = positions_full.shape[0]
    if indices is not None:
        positions = positions_full[indices]
    else:
        positions = positions_full

    if periodic:
        cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    else:
        cell = np.identity(3, dtype=np.float64)
    dtype = node_attrs.dtype
    cutoff = float(model.r_max.detach())
    edgeIndex, shifts, _, _ = get_neighborhood(positions, cutoff, [periodic, periodic, periodic], cell)
    cell_tensor = torch.tensor(cell, dtype=dtype, device=ptr.device)
    volume = torch.linalg.det(cell_tensor)
    if torch.abs(volume) > 0:
        rcell = 2 * torch.pi * torch.linalg.inv(cell_tensor.mT)
    else:
        rcell = torch.zeros((3, 3), dtype=dtype, device=ptr.device)
    inputDict = {
        "ptr": ptr,
        "node_attrs": node_attrs,
        "batch": batch,
        "pbc": pbc,
        "positions": torch.tensor(positions, dtype=dtype, device=ptr.device),
        "edge_index": torch.tensor(edgeIndex, dtype=torch.int64, device=ptr.device),
        "shifts": torch.tensor(shifts, dtype=dtype, device=ptr.device),
        "cell": cell_tensor,
        "rcell": rcell,
        "volume": volume.reshape(-1),
        "total_charge": charge,
        "total_spin": multiplicity,
        "external_field": torch.zeros((charge.shape[0], 3), dtype=dtype, device=ptr.device),
        "fermi_level": torch.zeros((1,), dtype=dtype, device=ptr.device)
    }
    # load mm position and charges
    if mmInfo is not None:
        mm_positions = positions_full[mmInfo["mm_atoms"]]
        inputDict["mm_positions"] = torch.tensor(
            mm_positions, dtype=dtype, device=ptr.device
        )
        inputDict["mm_charges"] = torch.tensor(
            mmInfo["mm_charges"], dtype=dtype, device=ptr.device
        )
        inputDict["mm_source_batch"] = torch.zeros(
            len(mmInfo["mm_atoms"]), dtype=torch.long, device=ptr.device
        )
    # eval and get results
    results = model(inputDict, compute_force=True)
    energy = float(results[returnEnergyType].detach())*energyScale
    forces = (results["forces"]*energyScale*lengthScale).detach().cpu().numpy()
    mm_forces = results.get("mm_forces")

    if mmInfo is not None and mm_forces is None:
        raise ValueError("The model returned no 'mm_forces' although MM charges were supplied; it does not implement electrostatic embedding.")
    if mm_forces is not None:
        mm_forces = (mm_forces * energyScale * lengthScale).detach().cpu().numpy()

    # Scatter ML and MM forces back to the full system.
    if indices is not None:
        f = np.zeros((numAtoms, 3), dtype=(np.float64 if dtype == torch.float64 else np.float32))
        f[indices] = forces
        if mmInfo is not None and mm_forces is not None:
            f[mmInfo["mm_atoms"]] += mm_forces.astype(f.dtype, copy=False)
        forces = f
    return energy, forces
