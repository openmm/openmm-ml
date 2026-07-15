import numpy as np
import openmm
import openmm.app
import os
import pytest

from openmmml import MLPotential

emle = pytest.importorskip("emle", reason="emle is not installed")
mace = pytest.importorskip("mace", reason="mace is not installed")
platform_ints = range(openmm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# TODO: EMLE energy values have a lot of numerical noise.  Is this expected?
atol = 0.05

@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestEMLEEmbedding:

    @pytest.mark.parametrize("periodic", (False, True))
    @pytest.mark.parametrize("interpolate", (False, True))
    def testEmbedding(self, platform_int, periodic, interpolate):
        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))

        subset = [atom.index for atom in pdb.topology.atoms() if atom.residue.chain.index == 0]

        mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
        ml_potential = MLPotential("mace-off23-small")
        mm_system = mm_force_field.createSystem(pdb.topology, nonbondedMethod=openmm.app.PME if periodic else openmm.app.NoCutoff)
        mixed_system = ml_potential.createMixedSystem(pdb.topology, mm_system, subset, embedding="emle", interpolate=interpolate)

        platform = openmm.Platform.getPlatform(platform_int)
        mm_context = openmm.Context(mm_system, openmm.VerletIntegrator(0.001), platform)
        mixed_context = openmm.Context(mixed_system, openmm.VerletIntegrator(0.001), platform)

        mm_context.setPositions(pdb.positions)
        mixed_context.setPositions(pdb.positions)

        mm_energy = mm_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

        # Reference energies are computed with EMLECalculator from EMLE-Engine
        if periodic:
            expected_energy = -33844.07052235671
        else:
            expected_energy = -33694.33837009875

        if interpolate:
            for lambda_value in (0.0, 0.25, 0.5, 0.75, 1.0):
                mixed_context.setParameter("lambda_interpolate", lambda_value)
                mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
                assert np.isclose(mixed_energy, expected_energy * lambda_value + mm_energy * (1 - lambda_value), rtol=0, atol=atol)

        else:
            mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            assert np.isclose(mixed_energy, expected_energy, rtol=0, atol=atol)
