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

atol = 0.02

@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestMechanicalEmbedding:
    def getTopologyPositionsSubset(self, topology, positions, subset):
        modeller = openmm.app.Modeller(topology, positions)
        modeller.delete([atom for atom in topology.atoms() if atom.index not in subset])
        return modeller.getTopology(), modeller.getPositions()

    def getBondedTerms(self, system):
        bonds = set()
        angles = set()
        torsions = set()
        cmaps = set()

        for force in system.getForces():
            if isinstance(force, openmm.HarmonicBondForce):
                for i in range(force.getNumBonds()):
                    bond = tuple(force.getBondParameters(i)[:2])
                    bonds.add(min(bond, bond[::-1]))
            elif isinstance(force, openmm.HarmonicAngleForce):
                for i in range(force.getNumAngles()):
                    angle = tuple(force.getAngleParameters(i)[:3])
                    angles.add(min(angle, angle[::-1]))
            elif isinstance(force, openmm.PeriodicTorsionForce):
                for i in range(force.getNumTorsions()):
                    torsion = tuple(force.getTorsionParameters(i)[:4])
                    torsions.add(min(torsion, torsion[::-1]))
            elif isinstance(force, openmm.CMAPTorsionForce):
                for i in range(force.getNumTorsions()):
                    cmap = tuple(force.getTorsionParameters(i)[1:])
                    cmaps.add((min(cmap[:4], cmap[:4][::-1]), min(cmap[4:], cmap[4:][::-1])))

        return bonds, angles, torsions, cmaps

    @pytest.mark.parametrize("periodic", (False, True))
    @pytest.mark.parametrize("interpolate", (False, True))
    @pytest.mark.parametrize("ff_family", ("amber", "charmm"))
    def testEmbedding(self, platform_int, periodic, interpolate, ff_family):
        """
        Mechanical embedding for a non-periodic system, or for a periodic
        long-range system (in both cases, all periodic images if any are present
        are included or excluded, so the verification calculation is the same).
        """

        if ff_family == "charmm" and interpolate:
            pytest.skip("Interpolation not yet supported with CustomNonbondedForce")

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        topology_ml_mm = pdb.topology
        positions_ml_mm = pdb.positions

        subset = [atom.index for atom in topology_ml_mm.atoms() if atom.residue.chain.index == 0]
        topology_ml, positions_ml = self.getTopologyPositionsSubset(topology_ml_mm, positions_ml_mm, set(subset))

        if ff_family == "amber":
            # Amber will result in an ordinary NonbondedForce only
            mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
        elif ff_family == "charmm":
            # CHARMM has NBFix and so a CustomNonbondedForce will also be used
            mm_force_field = openmm.app.ForceField("charmm36_2024.xml", "charmm36_2024/water.xml")
        else:
            raise NotImplementedError
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
            elif isinstance(force, openmm.CustomNonbondedForce):
                force.setUseLongRangeCorrection(False)

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
                assert np.isclose(mixed_energy, expected_energy * lambda_value + mm_energy_ml_mm * (1 - lambda_value), rtol=0, atol=atol)

        else:
            mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            assert np.isclose(mixed_energy, expected_energy, rtol=0, atol=atol)

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
        topology_ml, positions_ml = self.getTopologyPositionsSubset(topology_ml_mm, positions_ml_mm, set(subset))

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
                assert np.isclose(mixed_energy, expected_energy * lambda_value + mm_energy_ml_mm * (1 - lambda_value), rtol=0, atol=atol)

        else:
            mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            assert np.isclose(mixed_energy, expected_energy, rtol=0, atol=atol)

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

    @pytest.mark.parametrize("override_distance", (False, True))
    @pytest.mark.parametrize("ff_name", ("ethanol.xml", "ethanol_ljforce.xml"))
    def testLinkAtomTerms(self, platform_int, override_distance, ff_name):
        """
        Test for presence of the appropriate terms and positions of the virtual
        sites in the link-atom method.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "ethanol", "ethanol.pdb"))
        """
                  H4   H6
                  |    |
        H3 - O0 - C1 - C2 - H8
                  |    |
                  H5   H7
        """

        # Expected distances are in nanometers.
        expected_cc_distance = 0.1525970013793 # From force field.
        if override_distance:
            expected_ch_distance = 0.12
        else:
            expected_ch_distance = 0.107 # From default covalent radii.

        mm_force_field = openmm.app.ForceField(os.path.join(test_data_dir, "ethanol", ff_name))
        ml_potential = MLPotential("mace-off23-small")

        mm_system = mm_force_field.createSystem(pdb.topology)
        args = {}
        if override_distance:
            args["linkAtomDistances"] = [(1, 2, 0.12)]
        mixed_system = ml_potential.createMixedSystem(pdb.topology, mm_system, [0, 1, 3, 4, 5], interpolate=False, **args)

        # Get all of the bonded terms in both systems.
        mm_bonds, mm_angles, mm_torsions, _ = self.getBondedTerms(mm_system)
        mixed_bonds, mixed_angles, mixed_torsions, _ = self.getBondedTerms(mixed_system)

        # No bonded terms should be added to the mixed system.
        assert not mixed_bonds - mm_bonds
        assert not mixed_angles - mm_angles
        assert not mixed_torsions - mm_torsions

        # The appropriate terms should be removed from the mixed system.
        assert mm_bonds - mixed_bonds == {(0, 1), (0, 3), (1, 4), (1, 5)}
        assert mm_angles - mixed_angles == {(0, 1, 2), (0, 1, 4), (0, 1, 5), (1, 0, 3), (2, 1, 4), (2, 1, 5), (4, 1, 5)}
        assert mm_torsions - mixed_torsions == {(2, 1, 0, 3), (3, 0, 1, 4), (3, 0, 1, 5)}

        platform = openmm.Platform.getPlatform(platform_int)
        context = openmm.Context(mixed_system, openmm.LangevinIntegrator(300, 1, 0.001), platform)
        context.setPositions(pdb.positions + [openmm.Vec3(0, 0, 0)] * openmm.unit.nanometer)
        context.computeVirtualSites()

        def check_positions():
            positions = context.getState(positions=True).getPositions(asNumpy=True) / openmm.unit.nanometer
            delta_c1_c2 = positions[2] - positions[1]
            delta_c1_vs = positions[9] - positions[1]
            dist_c1_c2 = np.linalg.norm(delta_c1_c2)
            dist_c1_vs = np.linalg.norm(delta_c1_vs)

            # Virtual site should be the appropriate distance from C1.
            assert np.isclose(dist_c1_vs, expected_ch_distance)
            # Virtual site should be in line with C1-C2.
            assert np.isclose(delta_c1_c2 @ delta_c1_vs, dist_c1_c2 * dist_c1_vs)
            # C1-C2 distance should be appropriate.
            assert dist_c1_c2 < 1.5 * expected_cc_distance

        # Check positions, run some dynamics, and check again.
        check_positions()
        openmm.LocalEnergyMinimizer.minimize(context)
        context.getIntegrator().step(1000)
        check_positions()

    def testLinkAtomForbidden(self, platform_int):
        """
        Ensure that multiple ML-MM bonds to the same MM atom are disallowed.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "ethanol", "ethanol.pdb"))
        mm_force_field = openmm.app.ForceField(os.path.join(test_data_dir, "ethanol", "ethanol.xml"))
        ml_potential = MLPotential("mace-off23-small")

        mm_system = mm_force_field.createSystem(pdb.topology)
        with pytest.raises(ValueError, match="Multiple link bonds to MM atom 1"):
            ml_potential.createMixedSystem(pdb.topology, mm_system, [0, 3, 4, 5])

    def testLinkAtomMultipleRegions(self, platform_int):
        """
        Check that the correct bonded terms are present in a molecule with
        multiple ML and MM subregions.
        """

        topology = openmm.app.Topology()
        chain = topology.addChain()
        atoms = [topology.addAtom("X", openmm.app.element.carbon, topology.addResidue("X", chain)) for _ in range(22)]
        for pair in zip(atoms[:-1], atoms[1:]):
            topology.addBond(*pair)

        mm_system = openmm.System()

        bond_force = openmm.HarmonicBondForce()
        for i in range(len(atoms) - 1):
            bond_force.addBond(i, i + 1, 1, 1)
        mm_system.addForce(bond_force)

        angle_force = openmm.HarmonicAngleForce()
        for i in range(len(atoms) - 2):
            angle_force.addAngle(i, i + 1, i + 2, 1, 1)
        mm_system.addForce(angle_force)

        torsion_force = openmm.PeriodicTorsionForce()
        for i in range(len(atoms) - 3):
            torsion_force.addTorsion(i, i + 1, i + 2, i + 3, 1, 0, 1)
        mm_system.addForce(torsion_force)

        cmap_force = openmm.CMAPTorsionForce()
        for i in range(len(atoms) - 4):
            cmap_force.addTorsion(0, i, i + 1, i + 2, i + 3, i + 1, i + 2, i + 3, i + 4)
        mm_system.addForce(cmap_force)

        """
        The following should cover all the possible configurations of up to
        five atoms forward or in reverse (excluding forbidden ML-MM-ML).

             0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
            ML-ML-ML-ML-ML-MM-MM-MM-MM-MM-ML-ML-ML-MM-MM-MM-ML-MM-MM-ML-ML-MM
        """
        mixed_system = MLPotential("mace-off23-small").createMixedSystem(topology, mm_system, [0, 1, 2, 3, 4, 10, 11, 12, 16, 19, 20])
        mixed_bonds, mixed_angles, mixed_torsions, mixed_cmaps = self.getBondedTerms(mixed_system)

        assert mixed_bonds == {(i, i + 1) for i in [4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 20]}
        assert mixed_angles == {(i, i + 1, i + 2) for i in [4, 5, 6, 7, 8, 12, 13, 14, 16, 17]}
        assert mixed_torsions == {(i, i + 1, i + 2, i + 3) for i in [3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17]}
        assert mixed_cmaps == {((i, i + 1, i + 2, i + 3), (i + 1, i + 2, i + 3, i + 4)) for i in [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]}

    def testLinkAtomImpropers(self, platform_int):
        """
        Check that the correct improper terms are present in a molecule with
        multiple ML and MM subregions.
        """

        topology = openmm.app.Topology()
        chain = topology.addChain()
        atoms = [topology.addAtom("X", openmm.app.element.carbon, topology.addResidue("X", chain)) for _ in range(8)]
        topology.addBond(atoms[0], atoms[1])
        topology.addBond(atoms[1], atoms[2])
        topology.addBond(atoms[2], atoms[3])
        topology.addBond(atoms[3], atoms[4])
        topology.addBond(atoms[1], atoms[5])
        topology.addBond(atoms[2], atoms[6])
        topology.addBond(atoms[3], atoms[7])

        mm_system = openmm.System()
        torsion_force = openmm.PeriodicTorsionForce()
        torsion_force.addTorsion(0, 1, 2, 5, 1, 0, 1)
        torsion_force.addTorsion(1, 0, 2, 5, 1, 0, 1)
        torsion_force.addTorsion(2, 0, 1, 5, 1, 0, 1)
        torsion_force.addTorsion(5, 0, 1, 2, 1, 0, 1)
        torsion_force.addTorsion(1, 2, 3, 6, 1, 0, 1)
        torsion_force.addTorsion(2, 1, 3, 6, 1, 0, 1)
        torsion_force.addTorsion(3, 1, 2, 6, 1, 0, 1)
        torsion_force.addTorsion(6, 1, 2, 3, 1, 0, 1)
        torsion_force.addTorsion(2, 3, 4, 7, 1, 0, 1)
        torsion_force.addTorsion(3, 2, 4, 7, 1, 0, 1)
        torsion_force.addTorsion(4, 2, 3, 7, 1, 0, 1)
        torsion_force.addTorsion(7, 2, 3, 4, 1, 0, 1)
        mm_system.addForce(torsion_force)

        """
        An improper is added for each of the three improper centers and with
        the central atom as each of the four possible atoms.

              MM5         ML7
               |           |
        MM0 - MM1 - ML2 - ML3 - MM4
                     |
                    MM6
        """
        mixed_system = MLPotential("mace-off23-small").createMixedSystem(topology, mm_system, [2, 3, 7])
        _, _, mixed_torsions, _ = self.getBondedTerms(mixed_system)

        assert mixed_torsions == {(0, 1, 2, 5), (1, 0, 2, 5), (2, 0, 1, 5), (2, 1, 0, 5)}

    def testLinkAtomInterpolation(self, platform_int):
        """
        Ensure interpolation works as expected with the link-atom method.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "ethanol", "ethanol.pdb"))

        mm_force_field = openmm.app.ForceField(os.path.join(test_data_dir, "ethanol", "ethanol.xml"))
        ml_potential = MLPotential("mace-off23-small")

        mm_system = mm_force_field.createSystem(pdb.topology)
        mixed_system = ml_potential.createMixedSystem(pdb.topology, mm_system, [0, 1, 3, 4, 5], interpolate=False)
        interpolate_system = ml_potential.createMixedSystem(pdb.topology, mm_system, [0, 1, 3, 4, 5], interpolate=True)

        platform = openmm.Platform.getPlatform(platform_int)
        mm_context = openmm.Context(mm_system, openmm.VerletIntegrator(0.001), platform)
        mixed_context = openmm.Context(mixed_system, openmm.VerletIntegrator(0.001), platform)
        interpolate_context = openmm.Context(interpolate_system, openmm.VerletIntegrator(0.001), platform)

        mm_context.setPositions(pdb.positions)
        for context in (mixed_context, interpolate_context):
            context.setPositions(pdb.positions + [openmm.Vec3(0, 0, 0)] * openmm.unit.nanometer)
            context.computeVirtualSites()

        mm_energy = mm_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

        for lambda_value in (0.0, 0.25, 0.5, 0.75, 1.0):
            interpolate_context.setParameter("lambda_interpolate", lambda_value)
            interpolate_energy = interpolate_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            assert np.isclose(interpolate_energy, mixed_energy * lambda_value + mm_energy * (1 - lambda_value), rtol=0, atol=atol)

    def testLinkAtomInfo(self, platform_int):
        """
        Ensure the returnInfo keyword works with the link-atom method.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "ethanol", "ethanol.pdb"))
        mm_force_field = openmm.app.ForceField(os.path.join(test_data_dir, "ethanol", "ethanol.xml"))
        ml_potential = MLPotential("mace-off23-small")
        mm_system = mm_force_field.createSystem(pdb.topology)

        original_count = mm_system.getNumParticles()
        mixed_system = ml_potential.createMixedSystem(pdb.topology, mm_system, [0, 1, 3, 4, 5], returnInfo=False)
        mixed_info = ml_potential.createMixedSystem(pdb.topology, mm_system, [0, 1, 3, 4, 5], returnInfo=True)

        assert isinstance(mixed_system, openmm.System)
        assert isinstance(mixed_info["system"], openmm.System)
        assert isinstance(mixed_info["topology"], openmm.app.Topology)

        # Make sure the inputs were not modified.
        assert mm_system.getNumParticles() == pdb.topology.getNumAtoms() == original_count
        # Make sure the outputs have been modified and match.
        assert mixed_system.getNumParticles() == mixed_info["system"].getNumParticles() == mixed_info["topology"].getNumAtoms() > original_count
        # Make sure the virtual sites were appended to the end.
        assert mixed_info["oldToNew"] == list(range(original_count))
        for i in range(mixed_system.getNumParticles()):
            assert mixed_system.isVirtualSite(i) == (i >= original_count)
