# ruff: noqa: UP006, UP035, UP045

import os
import re
from typing import Dict, List, Optional

import metatomic.torch as mta
import numpy as np
import openmm as mm
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from openmm import app, unit

from openmmml import MLPotential

PLATFORMS = [mm.Platform.getPlatform(i) for i in range(mm.Platform.getNumPlatforms())]
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class HarmonicModel(torch.nn.Module):
    def __init__(self, force_constant, equilibrium_positions, requested=None):
        super().__init__()
        self.force_constant = force_constant
        self._requested = requested or {}
        self.register_buffer("equilibrium_positions", equilibrium_positions)

    def requested_inputs(self) -> dict[str, mta.ModelOutput]:
        return self._requested

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        result: dict[str, TensorMap] = {}
        for key in outputs:
            if key == "energy" or (len(key) >= 7 and key[0:7] == "energy/"):
                energy = torch.zeros(
                    (len(systems), 1), dtype=systems[0].positions.dtype
                )
                for i, system in enumerate(systems):
                    if selected_atoms is None:
                        pos = system.positions
                        eq = self.equilibrium_positions
                    else:
                        indices = selected_atoms.column("atom")
                        pos = system.positions[indices]
                        eq = self.equilibrium_positions[indices]
                    energy[i] += torch.sum(self.force_constant * (pos - eq) ** 2)

                block = TensorBlock(
                    values=energy,
                    samples=Labels(
                        "system", torch.arange(energy.shape[0]).reshape(-1, 1)
                    ),
                    components=[],
                    properties=Labels("energy", torch.tensor([[0]])),
                )
                result[key] = TensorMap(
                    keys=Labels("_", torch.tensor([[0]])), blocks=[block]
                )
            else:
                raise ValueError(f"unexpected output '{key}' requested")
        return result


class CustomInputAsEnergy(torch.nn.Module):
    """
    This model returns some custom input as the energy. It is used to test that the
    backend can pass extra inputs to the model.
    """

    def __init__(self, input_name, sample_kind="system"):
        super().__init__()
        self._input_name = input_name
        self._sample_kind = sample_kind

    def requested_inputs(self) -> dict[str, mta.ModelOutput]:
        input_unit = "e" if self._input_name == "charge" else ""
        return {
            self._input_name: mta.ModelOutput(
                unit=input_unit, sample_kind=self._sample_kind
            ),
        }

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        assert selected_atoms is None
        for key in outputs:
            if key != "energy":
                raise ValueError(f"unexpected output '{key}' requested")

        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            input_data = system.get_data(self._input_name).block().values
            # Touch positions so autograd can build forces. The energy is the input.
            energy[i] += input_data.reshape(()) + system.positions.sum() * 0

        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(energy.shape[0]).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {
            "energy": TensorMap(keys=Labels("_", torch.tensor([[0]])), blocks=[block])
        }


class WholeBoxEnergy(torch.nn.Module):
    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        assert selected_atoms is None
        for key in outputs:
            if key != "energy":
                raise ValueError(f"unexpected output '{key}' requested")
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            energy[i] += system.positions.sum()
        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(energy.shape[0]).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {
            "energy": TensorMap(keys=Labels("_", torch.tensor([[0]])), blocks=[block])
        }


class NeighborPairEnergy(torch.nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self._nl = mta.NeighborListOptions(cutoff=cutoff, full_list=True, strict=True)

    def requested_neighbor_lists(self) -> list[mta.NeighborListOptions]:
        return [self._nl]

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        assert selected_atoms is None
        for key in outputs:
            if key != "energy":
                raise ValueError(f"unexpected output '{key}' requested")
        dtype = systems[0].positions.dtype
        device = systems[0].positions.device
        energy = torch.zeros((len(systems), 1), dtype=dtype, device=device)
        for i, system in enumerate(systems):
            neighbors = system.get_neighbor_list(self._nl)
            energy[i] += neighbors.values.reshape(-1, 3).pow(2).sum()
        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(energy.shape[0]).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {
            "energy": TensorMap(keys=Labels("_", torch.tensor([[0]])), blocks=[block])
        }


class NeighborPairForce(torch.nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self._nl = mta.NeighborListOptions(cutoff=cutoff, full_list=True, strict=True)

    def requested_neighbor_lists(self) -> list[mta.NeighborListOptions]:
        return [self._nl]

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        dtype = systems[0].positions.dtype
        device = systems[0].positions.device
        energy = torch.zeros((len(systems), 1), dtype=dtype, device=device)
        all_forces = []
        for system_i, system in enumerate(systems):
            forces = torch.zeros((len(system), 3), dtype=dtype, device=device)
            neighbors = system.get_neighbor_list(self._nl)
            first = neighbors.samples.column("first_atom").to(torch.long)
            second = neighbors.samples.column("second_atom").to(torch.long)
            disp = neighbors.values.reshape(-1, 3)
            forces.index_add_(0, first, disp)
            forces.index_add_(0, second, -disp)
            if selected_atoms is not None:
                mask = selected_atoms.column("system") == system_i
                idx = selected_atoms.column("atom")[mask].to(torch.long)
                forces = forces[idx]
            all_forces.append(forces)

        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(energy.shape[0]).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        result = {
            "energy": TensorMap(keys=Labels("_", torch.tensor([[0]])), blocks=[block])
        }
        if "non_conservative_force" not in outputs:
            return result
        nc = torch.cat(all_forces).reshape(-1, 3, 1)
        if selected_atoms is None:
            rows = []
            for s, system in enumerate(systems):
                n_atoms = len(system)
                row = torch.zeros((n_atoms, 2), dtype=torch.int32, device=device)
                row[:, 0] = s
                row[:, 1] = torch.arange(n_atoms, device=device)
                rows.append(row)
            samples = Labels(["system", "atom"], torch.cat(rows))
        else:
            samples = selected_atoms
        result["non_conservative_force"] = TensorMap(
            keys=Labels("_", torch.tensor([[0]], device=device)),
            blocks=[
                TensorBlock(
                    values=nc,
                    samples=samples,
                    components=[
                        Labels(["xyz"], torch.arange(3, device=device).reshape(-1, 1))
                    ],
                    properties=Labels(
                        ["non_conservative_force"],
                        torch.tensor([[0]], device=device),
                    ),
                )
            ],
        )
        return result


def _atomistic_model(model, atomic_types, interaction_range=0.0, outputs=None):
    if outputs is None:
        outputs = {"energy": mta.ModelOutput(unit="kJ/mol", sample_kind="system")}

    capabilities = mta.ModelCapabilities(
        outputs=outputs,
        atomic_types=sorted(set(atomic_types)),
        interaction_range=interaction_range,
        length_unit="nm",
        supported_devices=["cpu"],
        dtype="float64",
    )

    return mta.AtomisticModel(model.eval(), mta.ModelMetadata(), capabilities)


def _energy(context, groups=None):
    if groups is None:
        state = context.getState(getEnergy=True)
    else:
        state = context.getState(getEnergy=True, groups=groups)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


def _forces(context, groups=None):
    if groups is None:
        state = context.getState(getForces=True)
    else:
        state = context.getState(getForces=True, groups=groups)
    return state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )


@pytest.fixture(scope="module")
def harmonic_toluene():
    pdb = app.PDBFile(os.path.join(TEST_DATA_DIR, "toluene", "toluene.pdb"))
    positions = np.asarray(pdb.getPositions(asNumpy=True), dtype=np.float64)
    numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
    harmonic_model = HarmonicModel(1.0, torch.tensor(positions, dtype=torch.float64))
    model = _atomistic_model(harmonic_model, numbers)
    yield pdb, numbers, positions, model


@pytest.mark.parametrize("platform", PLATFORMS, ids=lambda p: p.getName())
class TestMetatomicPotential:
    def testCreateMixedSystem(self, platform):
        prmtop = app.AmberPrmtopFile(
            os.path.join(TEST_DATA_DIR, "toluene", "toluene-explicit.prm7")
        )
        inpcrd = app.AmberInpcrdFile(
            os.path.join(TEST_DATA_DIR, "toluene", "toluene-explicit.rst7")
        )

        ml_atoms = list(range(15))
        positions = inpcrd.positions.value_in_unit(unit.nanometer)
        positions = np.asarray(positions, dtype=np.float64)[ml_atoms]

        numbers = [atom.element.atomic_number for atom in prmtop.topology.atoms()]
        model = _atomistic_model(
            HarmonicModel(1.0, torch.tensor(positions, dtype=torch.float64)),
            numbers,
        )

        mm_system = prmtop.createSystem(nonbondedMethod=app.PME)

        potential = MLPotential("metatomic", model=model, checkConsistency=True)
        mixed_system = potential.createMixedSystem(
            prmtop.topology, mm_system, ml_atoms, interpolate=False
        )
        interpolating_system = potential.createMixedSystem(
            prmtop.topology, mm_system, ml_atoms, interpolate=True
        )

        mm_context = mm.Context(mm_system, mm.VerletIntegrator(0.001), platform)
        mixed_context = mm.Context(mixed_system, mm.VerletIntegrator(0.001), platform)
        interpolating_context = mm.Context(
            interpolating_system, mm.VerletIntegrator(0.001), platform
        )
        mm_context.setPositions(inpcrd.positions)
        mixed_context.setPositions(inpcrd.positions)
        interpolating_context.setPositions(inpcrd.positions)

        assert np.isclose(
            _energy(mixed_context), _energy(interpolating_context), rtol=1e-5
        )
        interpolating_context.setParameter("lambda_interpolate", 0)
        assert np.isclose(
            _energy(mm_context), _energy(interpolating_context), rtol=1e-5
        )
        python_forces = [
            f for f in mixed_system.getForces() if isinstance(f, mm.PythonForce)
        ]
        assert python_forces
        assert not python_forces[0].usesPeriodicBoundaryConditions()
        assert len(python_forces[0].getParticles()) == 0

    def testSelectedAtoms(self, platform):
        prmtop = app.AmberPrmtopFile(
            os.path.join(TEST_DATA_DIR, "toluene", "toluene-explicit.prm7")
        )
        inpcrd = app.AmberInpcrdFile(
            os.path.join(TEST_DATA_DIR, "toluene", "toluene-explicit.rst7")
        )
        ml_atoms = list(range(15))
        positions = np.asarray(
            inpcrd.positions.value_in_unit(unit.nanometer), dtype=np.float64
        )
        numbers = [atom.element.atomic_number for atom in prmtop.topology.atoms()]
        delta = 0.01
        model = _atomistic_model(
            HarmonicModel(1.0, torch.tensor(positions[ml_atoms], dtype=torch.float64)),
            numbers,
        )

        potential = MLPotential("metatomic", model=model, checkConsistency=True)
        mixed = potential.createMixedSystem(
            prmtop.topology,
            prmtop.createSystem(nonbondedMethod=app.PME),
            ml_atoms,
            forceGroup=1,
        )

        context = mm.Context(mixed, mm.VerletIntegrator(0.001), platform)
        context.setPositions(positions + delta)
        energy = len(ml_atoms) * 3 * delta**2
        forces = np.zeros_like(positions)
        forces[ml_atoms] = -2 * delta
        assert np.isclose(energy, _energy(context, groups={1}), rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(
            forces, _forces(context, groups={1}), rtol=1e-4, atol=1e-5
        )

    def testExtraInputs(self, platform, harmonic_toluene):
        pdb, numbers, positions, _ = harmonic_toluene
        requested = {
            "charge": mta.ModelOutput(unit="e", sample_kind="system"),
            "spin_multiplicity": mta.ModelOutput(unit="", sample_kind="system"),
        }
        cases = [
            {},
            {"charge": 0, "multiplicity": 1},
            {"charge": 0, "spinMultiplicity": 1},
            {"charge": 0, "spin_multiplicity": 1},
        ]
        model = _atomistic_model(
            HarmonicModel(1.0, torch.tensor(positions, dtype=torch.float64), requested),
            numbers,
        )
        potential = MLPotential("metatomic", model=model, checkConsistency=True)
        for kwargs in cases:
            context = mm.Context(
                potential.createSystem(pdb.topology, **kwargs),
                mm.VerletIntegrator(0.001),
                platform,
            )
            context.setPositions(positions)
            assert np.isclose(_energy(context), 0.0, atol=1e-8)

        per_atom = _atomistic_model(
            CustomInputAsEnergy("charge", sample_kind="atom"), numbers
        )
        message = (
            "this model requests extra input 'charge' (sample_kind='atom'), "
            "which is not implemented by MLPotential('metatomic')"
        )
        with pytest.raises(ValueError, match=re.escape(message)):
            potential = MLPotential("metatomic", model=per_atom, checkConsistency=True)
            potential.createSystem(pdb.topology)

    def testPartialPbc(self, platform, harmonic_toluene):
        pdb, _, positions, model = harmonic_toluene
        box = [mm.Vec3(2, 0, 0), mm.Vec3(0, 2, 0), mm.Vec3(0, 0, 2)]

        potential = MLPotential("metatomic", model=model, checkConsistency=True)
        system = potential.createSystem(pdb.topology, pbc=(True, True, False))

        system.setDefaultPeriodicBoxVectors(*box)
        python_forces = [f for f in system.getForces() if isinstance(f, mm.PythonForce)]
        assert python_forces[0].usesPeriodicBoundaryConditions()

        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPeriodicBoxVectors(*box)
        context.setPositions(positions)
        assert np.isfinite(_energy(context))

    def testLennardJones(self, platform):
        import metatomic_lj_test

        # this model intentionally uses non-native units
        LJ_CUTOFF = 5.0  # in Angstrom
        LJ_SIGMA = 1.5808  # in Angstrom
        LJ_EPSILON = 0.1729  # in eV

        electron_volt = unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA

        pdb = app.PDBFile(os.path.join(TEST_DATA_DIR, "toluene", "toluene.pdb"))
        numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
        model = metatomic_lj_test.lennard_jones_model(
            atomic_type=numbers[0],
            cutoff=LJ_CUTOFF,
            sigma=LJ_SIGMA,
            epsilon=LJ_EPSILON,
            length_unit="Angstrom",
            energy_unit="eV",
            with_extension=False,
        )
        model._capabilities.atomic_types = sorted(set(numbers))
        positions = pdb.getPositions(asNumpy=True)

        # lj-test subtracts the pair energy at the cutoff. NonbondedForce does not.
        reference = mm.System()
        nonbonded = mm.NonbondedForce()
        nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
        nonbonded.setCutoffDistance(LJ_CUTOFF * unit.angstrom)
        nonbonded.setUseDispersionCorrection(False)
        for _ in numbers:
            reference.addParticle(1.0)
            nonbonded.addParticle(
                0.0,
                LJ_SIGMA * unit.angstrom,
                LJ_EPSILON * electron_volt,
            )
        reference.addForce(nonbonded)
        reference_context = mm.Context(
            reference,
            mm.VerletIntegrator(0.001),
            mm.Platform.getPlatformByName("Reference"),
        )
        reference_context.setPositions(positions)
        coords = np.asarray(positions.value_in_unit(unit.nanometer))
        cutoff_nm = (LJ_CUTOFF * unit.angstrom).value_in_unit(unit.nanometer)
        displacement = coords[:, None, :] - coords[None, :, :]
        distance2 = np.sum(displacement * displacement, axis=-1)
        n_pairs = int(np.sum(np.triu(distance2 <= cutoff_nm**2, k=1)))
        shift = (
            4
            * LJ_EPSILON
            * electron_volt
            * ((LJ_SIGMA / LJ_CUTOFF) ** 12 - (LJ_SIGMA / LJ_CUTOFF) ** 6)
        ).value_in_unit(unit.kilojoules_per_mole)
        energy_ref = _energy(reference_context) - n_pairs * shift
        forces_ref = _forces(reference_context)

        def run(**potential_kwargs):
            potential = MLPotential(
                "metatomic", model=model, checkConsistency=True, **potential_kwargs
            )
            system = potential.createSystem(pdb.topology, removeCMMotion=False)
            context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
            context.setPositions(positions)
            return context

        context = run(uncertaintyThreshold=None)
        assert np.isclose(energy_ref, _energy(context), rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(forces_ref, _forces(context), rtol=1e-4, atol=1e-5)

        context = run(variants={"energy": "doubled"}, uncertaintyThreshold=None)
        assert np.isclose(2.0 * energy_ref, _energy(context), rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(
            2.0 * forces_ref, _forces(context), rtol=1e-4, atol=1e-5
        )

        message = (
            "Some of the atomic energy uncertainties are larger than the "
            "threshold of 10.0 kJ/mol. The prediction is above the "
            f"threshold for atoms {list(range(len(numbers)))}."
        )
        with pytest.warns(UserWarning, match=re.escape(message)):
            _energy(run())


class TestMetatomicPotentialOptions:
    def testSpinMultiplicity(self, harmonic_toluene):
        pdb, numbers, positions, _ = harmonic_toluene
        model = _atomistic_model(CustomInputAsEnergy("spin_multiplicity"), numbers)
        potential = MLPotential("metatomic", model=model, checkConsistency=True)
        cases = [
            ({}, 1),
            ({"spin_multiplicity": 3}, 3),
            ({"multiplicity": 4}, 4),
            ({"spinMultiplicity": 5}, 5),
        ]
        for kwargs, spin in cases:
            context = mm.Context(
                potential.createSystem(pdb.topology, **kwargs),
                mm.VerletIntegrator(0.001),
            )
            context.setPositions(positions)
            assert np.isclose(_energy(context), spin, rtol=1e-5, atol=1e-8)

    def testCharge(self, harmonic_toluene):
        pdb, numbers, positions, _ = harmonic_toluene
        model = _atomistic_model(CustomInputAsEnergy("charge"), numbers)
        potential = MLPotential("metatomic", model=model, checkConsistency=True)
        for kwargs, charge in ({}, 0), ({"charge": 2}, 2), ({"charge": -1}, -1):
            context = mm.Context(
                potential.createSystem(pdb.topology, **kwargs),
                mm.VerletIntegrator(0.001),
            )
            context.setPositions(positions)
            assert np.isclose(_energy(context), charge, rtol=1e-5, atol=1e-8)

    def testNonConservativeRequiresOutput(self, harmonic_toluene):
        pdb, _, _, model = harmonic_toluene
        message = "output 'non_conservative_force' not found in outputs"
        with pytest.raises(ValueError, match=re.escape(message)):
            potential = MLPotential(
                "metatomic", model=model, checkConsistency=True, nonConservative=True
            )
            potential.createSystem(pdb.topology)

    def testAtomTypes(self, harmonic_toluene):
        pdb, numbers, positions, _ = harmonic_toluene
        custom_types = [100 + i for i in range(len(numbers))]
        model = _atomistic_model(
            HarmonicModel(1.0, torch.tensor(positions, dtype=torch.float64)),
            custom_types,
        )
        potential = MLPotential("metatomic", model=model, checkConsistency=True)
        message = "this model does not support atomic type 6"
        with pytest.raises(ValueError, match=re.escape(message)):
            potential.createSystem(pdb.topology)
        context = mm.Context(
            potential.createSystem(pdb.topology, atomTypes=custom_types),
            mm.VerletIntegrator(0.001),
        )
        context.setPositions(positions)
        assert np.isclose(_energy(context), 0.0, atol=1e-8)

        n_atoms = pdb.topology.getNumAtoms()
        atom_types = [6]
        message = (
            f"atomTypes must have length {n_atoms} (one entry per Topology "
            f"atom), got {len(atom_types)}"
        )
        with pytest.raises(ValueError, match=re.escape(message)):
            potential = MLPotential("metatomic", model=model, checkConsistency=True)
            potential.createSystem(pdb.topology, atomTypes=atom_types)

    def testInvalidPbcLength(self, harmonic_toluene):
        pdb, _, _, model = harmonic_toluene
        message = "pbc must be a length-3 sequence of booleans"
        with pytest.raises(ValueError, match=re.escape(message)):
            potential = MLPotential("metatomic", model=model, checkConsistency=True)
            potential.createSystem(pdb.topology, pbc=(True, False))

    def testAtomWithoutElement(self, harmonic_toluene):
        _, _, _, model = harmonic_toluene
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("X", chain)
        topology.addAtom("X", None, residue)
        message = (
            "All atoms in the Topology must have elements defined, or pass "
            "atomTypes with an integer type for every atom."
        )
        with pytest.raises(ValueError, match=re.escape(message)):
            potential = MLPotential("metatomic", model=model, checkConsistency=True)
            potential.createSystem(topology)

    def testInvalidNonConservative(self, harmonic_toluene):
        _, _, _, model = harmonic_toluene
        message = "nonConservative must be one of [True, False, 'forces'], got 'stress'"
        with pytest.raises(ValueError, match=re.escape(message)):
            MLPotential(
                "metatomic",
                model=model,
                checkConsistency=True,
                nonConservative="stress",
            )

    def testMechanicalEmbedding(self, harmonic_toluene):
        from openmmml.models.metatomicpotential import MetatomicPotentialImpl

        pdb, numbers, positions, _ = harmonic_toluene
        model = _atomistic_model(
            HarmonicModel(1.0, torch.tensor(positions, dtype=torch.float64)),
            numbers,
            interaction_range=0.5,
        )
        assert (
            MetatomicPotentialImpl(model, "cpu", None, True).getMLLongRange() is False
        )
        system = mm.System()
        for _ in numbers:
            system.addParticle(1.0)
        atoms = list(range(len(numbers)))
        potential = MLPotential("metatomic", model=model, checkConsistency=True)
        assert potential.createMixedSystem(pdb.topology, system, atoms) is not None
        message = "This ML model does not support the mlLongRange option."
        with pytest.raises(ValueError, match=re.escape(message)):
            potential.createMixedSystem(pdb.topology, system, atoms, mlLongRange=True)

    def testInfiniteRange(self, harmonic_toluene):
        from openmmml.models.metatomicpotential import MetatomicPotentialImpl

        pdb, numbers, positions, _ = harmonic_toluene
        model = _atomistic_model(
            HarmonicModel(1.0, torch.tensor(positions, dtype=torch.float64)),
            numbers,
            interaction_range=float("inf"),
        )
        assert (
            MetatomicPotentialImpl(model, "cpu", None, True).getMLLongRange() is False
        )
        system = mm.System()
        for _ in numbers:
            system.addParticle(1.0)
        message = (
            "this model has an infinite interaction range, which is not "
            "supported for mixed ML/MM systems"
        )
        potential = MLPotential("metatomic", model=model, checkConsistency=True)
        with pytest.raises(ValueError, match=re.escape(message)):
            potential.createMixedSystem(pdb.topology, system, list(range(len(numbers))))
        context = mm.Context(
            potential.createSystem(pdb.topology), mm.VerletIntegrator(0.001)
        )
        context.setPositions(positions)
        assert np.isclose(_energy(context), 0.0, atol=1e-8)


@pytest.mark.parametrize("platform", PLATFORMS, ids=lambda p: p.getName())
class TestMetatomicMixedRegion:
    def testTotalEnergyIsTheMlRegionOnly(self, platform):
        # Only the ML atoms are given to the model.
        #
        #   ML ------ ML                 MM
        #   (0, 0)   (0.1, 0)         (0.4, 0.2)
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.4, 0.2, 0.0]],
            dtype=np.float64,
        )
        ml_atoms = [0, 1]
        model = _atomistic_model(WholeBoxEnergy(), [6])
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("MOL", chain)
        system = mm.System()
        for _ in positions:
            topology.addAtom("C", app.element.carbon, residue)
            system.addParticle(1.0)

        potential = MLPotential("metatomic", model=model, checkConsistency=True)
        context = mm.Context(
            potential.createMixedSystem(topology, system, ml_atoms),
            mm.VerletIntegrator(0.001),
            platform,
        )
        context.setPositions(positions)
        expected = positions[ml_atoms].sum()
        assert not np.isclose(expected, positions.sum())
        np.testing.assert_allclose(_energy(context), expected, rtol=1e-5, atol=1e-8)
        forces = _forces(context)
        np.testing.assert_allclose(forces[ml_atoms], -1.0, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(forces[2], 0.0, atol=1e-8)

        moved = positions.copy()
        moved[2, 0] += 0.3
        context.setPositions(moved)
        np.testing.assert_allclose(_energy(context), expected, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(_forces(context)[2], 0.0, atol=1e-8)

    def testNeighborInsideCutoff(self, platform):
        # cutoff 0.5. MM is inside it, and is not given to the model.
        #
        #   ML -------- ML ---- MM
        #   0.0        0.2    0.35
        #   |<------ 0.5 ------>|
        cutoff = 0.5
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.35, 0.0, 0.0]],
            dtype=np.float64,
        )
        ml_atoms = [0, 1]
        model = _atomistic_model(
            NeighborPairEnergy(cutoff), [6], interaction_range=cutoff
        )
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("MOL", chain)
        system = mm.System()
        for _ in positions:
            topology.addAtom("C", app.element.carbon, residue)
            system.addParticle(1.0)

        potential = MLPotential("metatomic", model=model, checkConsistency=True)
        context = mm.Context(
            potential.createMixedSystem(topology, system, ml_atoms),
            mm.VerletIntegrator(0.001),
            platform,
        )
        context.setPositions(positions)
        ml_pair = 2.0 * 0.2**2
        extra = 2.0 * (0.35**2 + 0.15**2)
        assert not np.isclose(ml_pair, ml_pair + extra)
        np.testing.assert_allclose(_energy(context), ml_pair, rtol=1e-5, atol=1e-8)
        expected = np.zeros((3, 3))
        expected[0, 0] = 4.0 * 0.2
        expected[1, 0] = -4.0 * 0.2
        np.testing.assert_allclose(_forces(context), expected, rtol=1e-5, atol=1e-6)

    def testPeriodicImageOfMmAtom(self, platform):
        # 1 nm box, cutoff 0.3. The ML region is not periodic.
        #
        #   | ML                        MM | ML'
        #     0.05                    0.90   1.05
        #       <-------- 0.15 ------->
        cutoff = 0.3
        positions = np.array([[0.05, 0.5, 0.5], [0.90, 0.5, 0.5]], dtype=np.float64)
        box = [mm.Vec3(1, 0, 0), mm.Vec3(0, 1, 0), mm.Vec3(0, 0, 1)]
        model = _atomistic_model(
            NeighborPairEnergy(cutoff), [6], interaction_range=cutoff
        )
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("MOL", chain)
        system = mm.System()
        for _ in positions:
            topology.addAtom("C", app.element.carbon, residue)
            system.addParticle(1.0)
        topology.setPeriodicBoxVectors(box)
        system.setDefaultPeriodicBoxVectors(*box)
        potential = MLPotential("metatomic", model=model, checkConsistency=True)
        mixed = potential.createMixedSystem(topology, system, [0])
        python_forces = [f for f in mixed.getForces() if isinstance(f, mm.PythonForce)]
        context = mm.Context(mixed, mm.VerletIntegrator(0.001), platform)
        context.setPositions(positions)
        image_energy = 2.0 * 0.15**2
        assert image_energy > 0.0
        np.testing.assert_allclose(_energy(context), 0.0, atol=1e-8)
        assert not np.isclose(_energy(context), image_energy)
        np.testing.assert_allclose(_forces(context), 0.0, atol=1e-8)
        assert not python_forces[0].usesPeriodicBoundaryConditions()

    def testNonConservativeForcesStayOnTheRegion(self, platform):
        #   ML <-------> ML        MM
        #   0.0         0.2       0.35
        cutoff = 0.5
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.35, 0.0, 0.0]],
            dtype=np.float64,
        )
        outputs = {
            "energy": mta.ModelOutput(unit="kJ/mol", sample_kind="system"),
            "non_conservative_force": mta.ModelOutput(
                unit="kJ/mol/nm", sample_kind="atom"
            ),
        }
        model = _atomistic_model(
            NeighborPairForce(cutoff),
            [6],
            interaction_range=cutoff,
            outputs=outputs,
        )
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("MOL", chain)
        system = mm.System()
        for _ in positions:
            topology.addAtom("C", app.element.carbon, residue)
            system.addParticle(1.0)

        potential = MLPotential(
            "metatomic", model=model, checkConsistency=True, nonConservative=True
        )
        context = mm.Context(
            potential.createMixedSystem(topology, system, [0, 1]),
            mm.VerletIntegrator(0.001),
            platform,
        )
        context.setPositions(positions)
        expected = np.zeros((3, 3))
        expected[0, 0] = 2.0 * 0.2
        expected[1, 0] = -2.0 * 0.2
        np.testing.assert_allclose(_energy(context), 0.0, atol=1e-8)
        np.testing.assert_allclose(_forces(context), expected, rtol=1e-5, atol=1e-6)

    def testBondAcrossTheBoundary(self, platform):
        #   C(ML) ---- H(cap) · · · C(MM)
        #
        # H lies on the cut bond. Moving C(MM) off the axis moves H.
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("CC", chain)
        carbon_ml = topology.addAtom("C1", app.element.carbon, residue)
        carbon_mm = topology.addAtom("C2", app.element.carbon, residue)
        topology.addBond(carbon_ml, carbon_mm)
        system = mm.System()
        system.addParticle(12.0)
        system.addParticle(12.0)
        ml_atom = 0
        cap = 2
        model = _atomistic_model(WholeBoxEnergy(), [6, 1])
        potentials = MLPotential("metatomic", model=model, checkConsistency=True)
        mixed = potentials.createMixedSystem(topology, system, [ml_atom], forceGroup=1)
        assert mixed.getNumParticles() == 3
        assert mixed.isVirtualSite(cap)
        context = mm.Context(mixed, mm.VerletIntegrator(0.001), platform)

        def seen(real_positions):
            context.setPositions(list(real_positions) + [mm.Vec3(0, 0, 0)])
            context.computeVirtualSites()
            coords = np.asarray(
                context.getState(getPositions=True)
                .getPositions(asNumpy=True)
                .value_in_unit(unit.nanometer)
            )
            total = coords[ml_atom].sum() + coords[cap].sum()
            assert not np.isclose(total, coords[ml_atom].sum())
            assert not np.isclose(total, coords.sum())
            np.testing.assert_allclose(
                _energy(context, groups={1}), total, rtol=1e-5, atol=1e-8
            )
            return total

        first = seen([mm.Vec3(0.1, 0.2, 0.3), mm.Vec3(0.25, 0.2, 0.3)])
        moved = seen([mm.Vec3(0.1, 0.2, 0.3), mm.Vec3(0.25, 0.5, 0.3)])
        assert not np.isclose(first, moved)
