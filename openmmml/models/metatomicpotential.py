"""
metatomicpotential.py: Implements OpenMM-ML potentials from metatomic models.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.
Authors: Eric D. Boittier, Guillaume Fraux

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

import math
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import openmm

from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory

_INPUT_DEFAULTS = {"charge": (0.0, "e"), "spin_multiplicity": (1.0, "")}
_VALID_NC = (True, False, "forces")
_DEFAULT_UNCERTAINTY_THRESHOLD_KJ_MOL = 10.0
# Neighbor-list skin in nm (2 Å). Set to -1 once vesin supports auto skin.
_NL_SKIN = 0.2


class MetatomicPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates MetatomicPotentialImpl objects."""

    def createImpl(
        self,
        name,
        model,
        device=None,
        extensionsDirectory=None,
        checkConsistency: bool = False,
        nonConservative=False,
        variants=None,
        uncertaintyThreshold=_DEFAULT_UNCERTAINTY_THRESHOLD_KJ_MOL,
        **args,
    ) -> MLPotentialImpl:
        assert name == "metatomic"
        return MetatomicPotentialImpl(
            model,
            device,
            extensionsDirectory,
            checkConsistency,
            nonConservative,
            variants,
            uncertaintyThreshold,
        )


class MetatomicPotentialImpl(MLPotentialImpl):
    """This MLPotentialImpl evaluates a metatomic model.

    Load a TorchScript model produced by :func:`metatomic.torch.load_atomistic_model`
    (typically a ``.pt`` file) and install a single :class:`openmm.PythonForce`.
    Neighbor lists use ``vesin.metatomic`` (CPU and CUDA). Conservative forces are
    ``-dE/dx`` via autograd. :class:`openmm.PythonForce` returns only
    energy and forces, so an explicit virial cannot be passed to the integrator.

    >>> potential = MLPotential(
    ...     "metatomic",
    ...     model="model.pt",
    ...     device="cuda",
    ...     extensionsDirectory="./extensions",
    ...     checkConsistency=False,
    ...     nonConservative=False,
    ...     variants={"energy": "pbe"},
    ...     uncertaintyThreshold=10,
    ... )
    >>> system = potential.createSystem(topology)

    Optional ``createSystem()`` / ``createMixedSystem()`` arguments:

    - ``charge``: total charge (default 0), used if the model requests it
    - ``multiplicity``: spin multiplicity (default 1); ``spinMultiplicity`` and
      ``spin_multiplicity`` are accepted as aliases
    - ``atomTypes``: integer type for each Topology atom; defaults to element
      atomic numbers when omitted
    - ``pbc``: length-3 sequence of booleans; default is all-on or all-off from
      the topology and System

    Mixed ML/MM systems use mechanical embedding. A model with an infinite
    ``interaction_range`` is rejected there. ``createSystem()`` still accepts it.

    A subset of atoms (mechanical embedding) is evaluated on its own, as an
    isolated non-periodic molecule. Passing that subset as ``selected_atoms``
    would still let the model use every atom inside its cutoff, on top of the
    force-field terms between the subset and the rest of the system.
    """

    def __init__(
        self,
        model,
        device,
        extensionsDirectory,
        checkConsistency,
        nonConservative=False,
        variants=None,
        uncertaintyThreshold=_DEFAULT_UNCERTAINTY_THRESHOLD_KJ_MOL,
    ):
        if nonConservative not in _VALID_NC:
            raise ValueError(
                f"nonConservative must be one of {list(_VALID_NC)}, "
                f"got {nonConservative!r}"
            )

        self.model = model
        self.device = device
        self.extensionsDirectory = extensionsDirectory
        self.checkConsistency = checkConsistency
        self.nonConservative = nonConservative
        self.variants = variants
        self.uncertaintyThreshold = uncertaintyThreshold

    def getMLLongRange(self) -> bool:
        return False

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Iterable[int] | None,
        forceGroup: int,
        **args,
    ):
        try:
            import torch
            from metatomic.torch import (
                AtomisticModel,
                ModelEvaluationOptions,
                ModelOutput,
                load_atomistic_model,
                pick_device,
                pick_output,
            )
        except ImportError as e:
            raise ImportError(
                "Failed to import metatomic. Install it with "
                "'pip install metatomic-torch'."
            ) from e

        if "mlLongRange" in args:
            raise ValueError("This ML model does not support the mlLongRange option.")

        topology_atoms = list(topology.atoms())
        types = _resolve_atom_types(topology_atoms, args.get("atomTypes"))

        if isinstance(self.model, (str, Path)):
            model = load_atomistic_model(
                self.model, extensions_directory=self.extensionsDirectory
            )
        else:
            assert isinstance(self.model, AtomisticModel)
            model = self.model

        capabilities = model.capabilities()
        if atoms is not None and math.isinf(capabilities.interaction_range):
            raise ValueError(
                "this model has an infinite interaction range, which is not "
                "supported for mixed ML/MM systems"
            )
        allowed = set(capabilities.atomic_types)
        for atom_type in types:
            if atom_type not in allowed:
                raise ValueError(f"this model does not support atomic type {atom_type}")
        desired = self.device
        if desired is not None and not isinstance(desired, str):
            desired = str(desired)
        device = torch.device(pick_device(capabilities.supported_devices, desired))
        dtype = getattr(torch, capabilities.dtype)
        model = model.to(device=device)
        types = torch.tensor(types, dtype=torch.int32, device=device)

        energy_key, nc_forces_key, uq_key = _resolve_output_keys(
            capabilities.outputs,
            self.variants,
            self.nonConservative,
            self.uncertaintyThreshold,
            pick_output,
        )

        extras = _extra_inputs(
            model.requested_inputs(use_new_names=True), args, dtype, device
        )

        neighbor_lists = []
        requested_nl = model.requested_neighbor_lists()
        if requested_nl:
            try:
                import vesin.metatomic
            except ImportError as e:
                raise ImportError(
                    "Failed to import vesin. Install it with 'pip install vesin'."
                ) from e
            neighbor_lists = [
                vesin.metatomic.NeighborList(
                    options=options,
                    length_unit="nm",
                    check_consistency=self.checkConsistency,
                    skin=_NL_SKIN,
                )
                for options in requested_nl
            ]

        # With a subset of atoms, the model sees only those atoms (and the link
        # caps among them), as an isolated, non-periodic molecule.
        region = None
        pbc = _resolve_pbc(args, topology, system, device)
        if atoms is not None:
            region = torch.tensor(list(atoms), dtype=torch.long, device=device)
            pbc = torch.zeros(3, dtype=torch.bool, device=device)
        outputs = {
            energy_key: ModelOutput(unit="kJ/mol", sample_kind="system"),
        }

        if nc_forces_key is not None:
            outputs[nc_forces_key] = ModelOutput(unit="kJ/mol/nm", sample_kind="atom")
        if uq_key is not None:
            outputs[uq_key] = ModelOutput(unit="kJ/mol", sample_kind="atom")
        evaluation_options = ModelEvaluationOptions(
            length_unit="nm",
            outputs=outputs,
        )

        compute = _ComputeMetatomic(
            model=model,
            types=types,
            extras=extras,
            neighbor_lists=neighbor_lists,
            evaluation_options=evaluation_options,
            energy_key=energy_key,
            nc_forces_key=nc_forces_key,
            uq_key=uq_key,
            uncertainty_threshold=self.uncertaintyThreshold,
            check_consistency=self.checkConsistency,
            pbc=pbc,
            dtype=dtype,
            region=region,
        )
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(bool(pbc.any()))
        system.addForce(force)


def _resolve_atom_types(topology_atoms, atom_types):
    n_atoms = len(topology_atoms)
    if atom_types is not None:
        atom_types = list(atom_types)
        if len(atom_types) != n_atoms:
            raise ValueError(
                f"atomTypes must have length {n_atoms} (one entry per Topology "
                f"atom), got {len(atom_types)}"
            )
        return [int(t) for t in atom_types]
    if any(atom.element is None for atom in topology_atoms):
        raise ValueError(
            "All atoms in the Topology must have elements defined, or pass "
            "atomTypes with an integer type for every atom."
        )
    return [atom.element.atomic_number for atom in topology_atoms]


def _resolve_output_keys(
    outputs, variants, non_conservative, uncertainty_threshold, pick_output
):
    variants = dict(variants or {})
    default_variant = variants.get("energy")
    resolved = {
        key: variants.get(key, default_variant)
        for key in [
            "energy",
            "energy_uncertainty",
            "non_conservative_force",
        ]
    }

    energy_key = pick_output("energy", outputs, resolved["energy"])

    has_energy_uq = any("energy_uncertainty" in key for key in outputs)
    uq_key = (
        pick_output("energy_uncertainty", outputs, resolved["energy_uncertainty"])
        if has_energy_uq and uncertainty_threshold is not None
        else None
    )

    nc_forces = non_conservative in (True, "forces")
    nc_forces_key = (
        pick_output(
            "non_conservative_force",
            outputs,
            resolved["non_conservative_force"],
        )
        if nc_forces
        else None
    )
    return energy_key, nc_forces_key, uq_key


def _resolve_pbc(args, topology, system, device):
    import torch

    periodic = (
        topology.getPeriodicBoxVectors() is not None
        or system.usesPeriodicBoundaryConditions()
    )
    user_pbc = args.get("pbc")
    if user_pbc is None:
        flags = [periodic, periodic, periodic]
    else:
        flags = [bool(x) for x in user_pbc]
        if len(flags) != 3:
            raise ValueError("pbc must be a length-3 sequence of booleans")
    return torch.tensor(flags, dtype=torch.bool, device=device)


def _extra_inputs(requested, args, dtype, device):
    import torch
    from metatensor.torch import Labels, TensorBlock, TensorMap

    extras = {}
    for name, option in requested.items():
        if name not in _INPUT_DEFAULTS or option.sample_kind != "system":
            raise ValueError(
                f"this model requests extra input '{name}' "
                f"(sample_kind={option.sample_kind!r}), which is not "
                "implemented by MLPotential('metatomic')"
            )
        default, input_unit = _INPUT_DEFAULTS[name]

        if name == "spin_multiplicity":
            value = default
            for aliases in ("multiplicity", "spinMultiplicity", "spin_multiplicity"):
                if aliases in args:
                    value = args[aliases]
        else:
            value = args.get(name, default)

        block = TensorBlock(
            values=torch.tensor([[float(value)]], dtype=dtype),
            samples=Labels(["system"], torch.zeros((1, 1), dtype=torch.int32)),
            components=[],
            properties=Labels([name], torch.tensor([[0]])),
        )
        tensor = TensorMap(Labels(["_"], torch.tensor([[0]])), [block])
        tensor.set_info("unit", input_unit)
        extras[name] = tensor.to(dtype=dtype, device=device)
    return extras


class _ComputeMetatomic:
    def __init__(
        self,
        model,
        types,
        pbc,
        extras,
        neighbor_lists,
        evaluation_options,
        energy_key,
        nc_forces_key,
        uq_key,
        uncertainty_threshold,
        check_consistency,
        dtype,
        region=None,
    ):
        self.model = model

        self.pbc = pbc
        self.types = types
        self.extras = extras
        self.neighbor_lists = neighbor_lists

        self.evaluation_options = evaluation_options
        self.energy_key = energy_key
        self.nc_forces_key = nc_forces_key
        self.uq_key = uq_key
        self.uncertainty_threshold = uncertainty_threshold
        self.check_consistency = check_consistency

        self.dtype = dtype
        self.region = region

    def __call__(self, state):
        import torch
        from metatomic.torch import System

        positions = np.asarray(state.getPositions(asNumpy=True), dtype=np.float64)
        device = self.types.device
        positions = torch.tensor(positions, dtype=self.dtype, device=device)
        types = self.types

        if self.region is not None:
            positions, types = positions[self.region], types[self.region]
        if bool(self.pbc.any()):
            cell = torch.tensor(
                np.asarray(state.getPeriodicBoxVectors(asNumpy=True), dtype=np.float64),
                dtype=self.dtype,
                device=device,
            )
            cell = cell * self.pbc.to(dtype=self.dtype).unsqueeze(1)
        else:
            cell = torch.zeros((3, 3), dtype=self.dtype, device=device)

        do_force_grad = self.nc_forces_key is None
        if do_force_grad:
            positions.requires_grad_(True)

        system = System(types, positions, cell, self.pbc)
        for name, tensor in self.extras.items():
            system.add_data(name, tensor)

        if self.neighbor_lists:
            if system.device.type not in ("cpu", "cuda"):
                system = system.to(device="cpu")
            for neighbors in self.neighbor_lists:
                neighbors.add_neighbor_list(systems=[system], copy=False)
            if system.device != device:
                system = system.to(device=device)

        outputs = self.model([system], self.evaluation_options, self.check_consistency)
        energy = outputs[self.energy_key].block().values

        if self.uq_key is not None:
            block = outputs[self.uq_key].block()
            uncertainty = block.values.detach().cpu().numpy().reshape(-1)
            above = np.flatnonzero(uncertainty > self.uncertainty_threshold)
            if len(above):
                atoms = block.samples.column("atom").cpu().numpy()
                if self.region is not None:
                    atoms = self.region.cpu().numpy()[atoms]
                flagged = sorted(int(i) for i in atoms[above])
                warnings.warn(
                    "Some of the atomic energy uncertainties are larger than the "
                    f"threshold of {self.uncertainty_threshold} kJ/mol. The "
                    f"prediction is above the threshold for atoms {flagged}.",
                    stacklevel=2,
                )

        if do_force_grad:
            energy.backward(-torch.ones_like(energy))

        if self.nc_forces_key is not None:
            block = outputs[self.nc_forces_key].block()
            nc_forces = block.values.detach().reshape(-1, 3)
            # remove the mean force to prevent drift since non-conservative forces
            # are not guaranteed to sum to zero.
            nc_forces = nc_forces - nc_forces.mean(dim=0, keepdim=True)
            atoms = block.samples.column("atom").detach().cpu().numpy()
            if self.region is not None:
                atoms = self.region.cpu().numpy()[atoms]
            forces = np.zeros((len(self.types), 3), dtype=np.float64)
            forces[atoms] = nc_forces.cpu().numpy()
        else:
            grad = system.positions.grad
            assert grad is not None

            if self.region is not None:
                forces = np.zeros((len(self.types), 3), dtype=np.float64)
                forces[self.region.cpu().numpy()] = grad.cpu().numpy()
            else:
                forces = grad.cpu().numpy()
        return float(energy.detach()), forces
