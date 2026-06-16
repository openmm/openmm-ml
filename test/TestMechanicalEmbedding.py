import numpy as np
import openmm
import openmm.app
import os
import pytest

from openmmml import MLPotential

ase = pytest.importorskip("ase", reason="ase is not installed")
mace = pytest.importorskip("mace", reason="mace is not installed")
platform_ints = range(openmm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestMechanicalEmbedding:

    def getTopologySubset(self, old_topology, subset):
        new_topology = openmm.app.Topology()
        new_topology.setPeriodicBoxVectors(old_topology.getPeriodicBoxVectors())

        new_atoms = {}

        for old_chain in old_topology.chains():
            if not any(atom.index in subset for atom in old_chain.atoms()):
                continue
            new_chain = new_topology.addChain(old_chain.id)

            for old_residue in old_chain.residues():
                if not any(atom.index in subset for atom in old_residue.atoms()):
                    continue
                new_residue = new_topology.addResidue(old_residue.name, new_chain, old_residue.id, old_residue.insertionCode)

                for old_atom in old_residue.atoms():
                    if old_atom.index in subset:
                        new_atoms[old_atom.index] = new_topology.addAtom(old_atom.name, old_atom.element, new_residue, old_atom.id, old_atom.formalCharge)

        for old_bond in old_topology.bonds():
            if old_bond.atom1.index in new_atoms and old_bond.atom2.index in new_atoms:
                new_topology.addBond(new_atoms[old_bond.atom1.index], new_atoms[old_bond.atom2.index], old_bond.order)

        return new_topology

    @pytest.mark.parametrize("periodic", (False, True))
    @pytest.mark.parametrize("interpolate", (False, True))
    def testEmbedding(self, platform_int, periodic, interpolate):
        """
        Mechanical embedding for a non-periodic system, or for a periodic
        long-range system (in both cases, all periodic images if any are present
        are included or excluded, so the verification calculation is the same).
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        topology_ml_mm = pdb.topology
        positions_ml_mm = pdb.positions

        subset = [atom.index for atom in topology_ml_mm.atoms() if atom.residue.chain.index == 0]
        topology_ml = self.getTopologySubset(topology_ml_mm, set(subset))
        positions_ml = [positions_ml_mm[index] for index in subset]

        mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
        ml_potential = MLPotential("ase")

        from mace.calculators.foundations_models import mace_off

        mm_system_ml_mm = mm_force_field.createSystem(topology_ml_mm, nonbondedMethod=openmm.app.PME if periodic else openmm.app.NoCutoff)
        mm_system_ml = mm_force_field.createSystem(topology_ml, nonbondedMethod=openmm.app.PME if periodic else openmm.app.NoCutoff)
        ml_system_ml = ml_potential.createSystem(topology_ml, calculator=mace_off("small"))
        mixed_system = ml_potential.createMixedSystem(topology_ml_mm, mm_system_ml_mm, subset, embedding="mechanical", interpolate=interpolate, calculator=mace_off("small"), mlLongRange=periodic)

        # Disable the dispersion correction for this system for the test so that
        # the same dispersion correction contributions are present on both sides
        # of the energy comparison.
        for force in mm_system_ml.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setUseDispersionCorrection(False)

        platform = openmm.Platform.getPlatform(platform_int)
        mm_context_ml_mm = openmm.Context(mm_system_ml_mm, openmm.VerletIntegrator(0.001), platform)
        mm_context_ml = openmm.Context(mm_system_ml, openmm.VerletIntegrator(0.001), platform)
        ml_context_ml = openmm.Context(ml_system_ml, openmm.VerletIntegrator(0.001), platform)
        mixed_context = openmm.Context(mixed_system, openmm.VerletIntegrator(0.001), platform)

        mm_context_ml_mm.setPositions(positions_ml_mm)
        mm_context_ml.setPositions(positions_ml)
        ml_context_ml.setPositions(positions_ml)
        mixed_context.setPositions(positions_ml_mm)

        mm_energy_ml_mm = mm_context_ml_mm.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        mm_energy_ml = mm_context_ml.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        ml_energy_ml = ml_context_ml.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

        # This is the standard expression for mechanical embedding.
        expected_energy = mm_energy_ml_mm - mm_energy_ml + ml_energy_ml

        if interpolate:
            for lambda_value in (0.0, 0.25, 0.5, 0.75, 1.0):
                mixed_context.setParameter("lambda_interpolate", lambda_value)
                mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
                assert np.isclose(mixed_energy, expected_energy * lambda_value + mm_energy_ml_mm * (1 - lambda_value), rtol=0, atol=1e-3)

        else:
            mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            assert np.isclose(mixed_energy, expected_energy, rtol=0, atol=1e-3)

    @pytest.mark.parametrize("interpolate", (False, True))
    def testPeriodicShortRange(self, platform_int, interpolate):
        """
        Mechanical embedding for a periodic system where the ML potential is
        assumed to not include interactions with periodic images.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        topology_ml_mm = pdb.topology
        positions_ml_mm = pdb.positions

        subset = [atom.index for atom in topology_ml_mm.atoms() if atom.residue.chain.index == 0]
        topology_ml = self.getTopologySubset(topology_ml_mm, set(subset))
        positions_ml = [positions_ml_mm[index] for index in subset]

        mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
        ml_potential = MLPotential("mace-off23-small")

        # When we compute the MM energy of the ML subset to subtract for the
        # energy comparison, compute it without contributions from any of the
        # periodic images.
        mm_system_ml_mm = mm_force_field.createSystem(topology_ml_mm, nonbondedMethod=openmm.app.PME)
        mm_system_ml = mm_force_field.createSystem(topology_ml, nonbondedMethod=openmm.app.NoCutoff)
        ml_system_ml = ml_potential.createSystem(topology_ml)
        mixed_system = ml_potential.createMixedSystem(topology_ml_mm, mm_system_ml_mm, subset, embedding="mechanical", interpolate=interpolate)

        # Disable the dispersion correction for this system for the test so that
        # the same dispersion correction contributions are present on both sides
        # of the energy comparison.
        for force in mm_system_ml.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setUseDispersionCorrection(False)

        platform = openmm.Platform.getPlatform(platform_int)
        mm_context_ml_mm = openmm.Context(mm_system_ml_mm, openmm.VerletIntegrator(0.001), platform)
        mm_context_ml = openmm.Context(mm_system_ml, openmm.VerletIntegrator(0.001), platform)
        ml_context_ml = openmm.Context(ml_system_ml, openmm.VerletIntegrator(0.001), platform)
        mixed_context = openmm.Context(mixed_system, openmm.VerletIntegrator(0.001), platform)

        mm_context_ml_mm.setPositions(positions_ml_mm)
        mm_context_ml.setPositions(positions_ml)
        ml_context_ml.setPositions(positions_ml)
        mixed_context.setPositions(positions_ml_mm)

        mm_energy_ml_mm = mm_context_ml_mm.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        mm_energy_ml = mm_context_ml.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        ml_energy_ml = ml_context_ml.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

        expected_energy = mm_energy_ml_mm - mm_energy_ml + ml_energy_ml

        if interpolate:
            for lambda_value in (0.0, 0.25, 0.5, 0.75, 1.0):
                mixed_context.setParameter("lambda_interpolate", lambda_value)
                mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
                assert np.isclose(mixed_energy, expected_energy * lambda_value + mm_energy_ml_mm * (1 - lambda_value), rtol=0, atol=1e-3)

        else:
            mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            print(mixed_energy - expected_energy)
            assert np.isclose(mixed_energy, expected_energy, rtol=0, atol=1e-3)

    @pytest.mark.parametrize("periodic", (False, True))
    @pytest.mark.parametrize("long_range", (False, True, None))
    def testMLLongRangeUnknown(self, platform_int, periodic, long_range):
        """
        An error should be raised if we need to know whether the ML potential is
        long-range or not, and this is not reported or specified.  Check all of
        the cases to ensure this.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        topology_ml_mm = pdb.topology

        subset = [atom.index for atom in topology_ml_mm.atoms() if atom.residue.chain.index == 0]

        mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
        ml_potential = MLPotential("ase")

        from mace.calculators.foundations_models import mace_off

        mm_system_ml_mm = mm_force_field.createSystem(topology_ml_mm, nonbondedMethod=openmm.app.PME if periodic else openmm.app.NoCutoff)
        kwargs = dict(topology=topology_ml_mm, system=mm_system_ml_mm, atoms=subset, calculator=mace_off("small"), embedding="mechanical", mlLongRange=long_range)

        if periodic and long_range is None:
            with pytest.raises(ValueError, match="The system is periodic and it is unknown if the ML model uses long-range interactions"):
                ml_potential.createMixedSystem(**kwargs)
        else:
            ml_potential.createMixedSystem(**kwargs)

    @pytest.mark.parametrize("remove", (False, True))
    def testRemoveConstraints(self, platform_int, remove):
        """
        Constraints in the ML region should be removed if specified.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        topology_ml_mm = pdb.topology

        subset = [atom.index for atom in topology_ml_mm.atoms() if atom.residue.chain.index == 0]
        subset_set = set(subset)

        mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
        ml_potential = MLPotential("mace-off23-small")

        mm_system_ml_mm = mm_force_field.createSystem(topology_ml_mm, constraints=openmm.app.AllBonds)
        mixed_system = ml_potential.createMixedSystem(topology_ml_mm, mm_system_ml_mm, subset, removeConstraints=remove, embedding="mechanical")

        mm_constraints = set()
        for index in range(mm_system_ml_mm.getNumConstraints()):
            atom_1, atom_2, _ = mm_system_ml_mm.getConstraintParameters(index)
            mm_constraints.add((atom_1, atom_2))

        mixed_constraints = set()
        for index in range(mixed_system.getNumConstraints()):
            atom_1, atom_2, _ = mixed_system.getConstraintParameters(index)
            mixed_constraints.add((atom_1, atom_2))

        # Constraints should be removed only if removeConstraints is set, and
        # constraints should never be added.
        assert bool(mm_constraints - mixed_constraints) == remove
        assert not mixed_constraints - mm_constraints

        for bond in topology_ml_mm.bonds():
            atom_1 = bond.atom1.index
            atom_2 = bond.atom2.index

            assert (atom_1, atom_2) in mm_constraints or (atom_2, atom_1) in mm_constraints
            if atom_1 in subset_set and atom_2 in subset_set:
                assert ((atom_1, atom_2) in mixed_constraints or (atom_2, atom_1) in mixed_constraints) != remove
