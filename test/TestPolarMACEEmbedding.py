import numpy as np
import openmm
import openmm.app
import openmm.unit
import os
import pytest

from openmmml import MLPotential

mace = pytest.importorskip("mace", reason="mace is not installed")
pytest.importorskip("graph_longrange.external_source_features", reason="graph_longrange external sources are not installed")
platform_ints = range(openmm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

atol = 0.05
# Direct-sum and PME forces agree to ~1e-3 kJ/mol/nm on this system (CPU, OpenCL); allow 10x.
force_atol = 0.01

# Reference energies are computed with MACE and graph_longrange alone (test/data/polarmace_energies.py)
expected_energies = {False: -1326354.2727021126, True: -1326502.4336419594}


def createSystems(periodic, **args):
    pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
    subset = [atom.index for atom in pdb.topology.atoms() if atom.residue.chain.index == 0]
    mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    mm_system = mm_force_field.createSystem(pdb.topology, nonbondedMethod=openmm.app.PME if periodic else openmm.app.NoCutoff)
    mixed_system = MLPotential("mace-polar-1-small").createMixedSystem(
        pdb.topology, mm_system, subset, embedding="electrostatic", precision="double", **args
    )
    return pdb, mixed_system


def computeEnergy(pdb, system, platform_int):
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatform(platform_int))
    context.setPositions(pdb.positions)
    return context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)


def computeForces(pdb, system, platform_int):
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatform(platform_int))
    context.setPositions(pdb.positions)
    return context.getState(forces=True).getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)


@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestPolarMACEEmbedding:

    @pytest.mark.parametrize("periodic", (False, True))
    def testEmbedding(self, platform_int, periodic):
        pdb, mixed_system = createSystems(periodic)
        assert np.isclose(computeEnergy(pdb, mixed_system, platform_int), expected_energies[periodic], rtol=0, atol=atol)

    def testPME(self, platform_int):
        # The PME evaluation of the MM field must reproduce the direct sum, in the
        # energy and in the forces on both the ML and the MM atoms.
        pytest.importorskip("torchpme", reason="torch-pme is not installed")
        pdb, mixed_system = createSystems(True, mmFieldMethod="PME")
        assert np.isclose(computeEnergy(pdb, mixed_system, platform_int), expected_energies[True], rtol=0, atol=atol)
        _, direct_system = createSystems(True)
        forces_pme = computeForces(pdb, mixed_system, platform_int)
        forces_direct = computeForces(pdb, direct_system, platform_int)
        assert np.allclose(forces_pme, forces_direct, rtol=0, atol=force_atol)
