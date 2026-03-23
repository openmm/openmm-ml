import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml import MLPotential

pytest.importorskip("orb_models", reason="orb-models is not installed")
platform_ints = range(mm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestOrb:
    @pytest.mark.parametrize("model", [
        "orb-mptraj-only-v2",
        "orb-v2",
        "orb-d3-xs-v2",
        "orb-d3-sm-v2",
        "orb-d3-v2",
        "orb-v3-direct-20-mpa",
        "orb-v3-direct-inf-mpa",
        "orb-v3-direct-20-omat",
        "orb-v3-direct-inf-omat",
        "orb-v3-direct-omol",
        "orb-v3-conservative-20-mpa",
        "orb-v3-conservative-inf-mpa",
        "orb-v3-conservative-20-omat",
        "orb-v3-conservative-inf-omat",
        "orb-v3-conservative-omol",
        "orb-v3-omat",
        "orb-v3-omol",
    ])
    def testCreatePureMLSystem(self, platform_int, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        if model in ("orb-v3-omat", "orb-v3-omol"):
            potential = MLPotential(model)
        else:
            potential = MLPotential("orb", modelName=model)
        system = potential.createSystem(pdb.topology)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        energyML = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        # Reference energies are calculated with ORBCalculator
        refEnergy = {
            "orb-mptraj-only-v2": -8946.530869816459,
            "orb-v2": -8945.339820108089,
            "orb-d3-xs-v2": -8940.841362278761,
            "orb-d3-sm-v2": -8931.112002578187,
            "orb-d3-v2": -8930.495866233374,
            "orb-v3-direct-20-mpa": -8919.63581904698,
            "orb-v3-direct-inf-mpa": -8901.550708271181,
            "orb-v3-direct-20-omat": -8941.013615450429,
            "orb-v3-direct-inf-omat": -8907.524359929663,
            "orb-v3-direct-omol": -712903.2181193709,
            "orb-v3-conservative-20-mpa": -8940.646289242472,
            "orb-v3-conservative-inf-mpa": -8917.839674863775,
            "orb-v3-conservative-20-omat": -8934.64760934537,
            "orb-v3-conservative-inf-omat": -8909.405894574036,
            "orb-v3-conservative-omol": -712903.547903221,
            "orb-v3-omat": -8909.405894574036,
            "orb-v3-omol": -712903.547903221,
        }
        assert np.isclose(refEnergy[model], energyML, rtol=1e-5)

    def testOverrideChargeSpin(self, platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential("orb-v3-omol")
        system = potential.createSystem(pdb.topology, charge=-1, multiplicity=3)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        energyML = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        energyRef = -712892.4765882556 # Calculated with ORBCalculator
        assert np.isclose(energyRef, energyML, rtol=1e-5)

    def testPeriodicSystem(self, platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        potential = MLPotential("orb-v3-omol")
        system = potential.createSystem(pdb.topology)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        positionsOriginal = pdb.getPositions(asNumpy=True)
        energyRef = -151632910.67503712 # Calculated with ORBCalculator
        for i in range(3):
            positions = positionsOriginal + i * 0.9 * unit.nanometers # translate molecule to test PBC
            context.setPositions(positions)
            energyML = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            assert np.isclose(energyRef, energyML, rtol=1e-5)

    def testCreateMixedSystem(self, platform_int):
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))
        mlAtoms = list(range(15))
        mmSystem = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential("orb-v3-omol")
        mixedSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, interpolate=False)
        interpSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, interpolate=True)
        platform = mm.Platform.getPlatform(platform_int)
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), platform)
        mixedContext = mm.Context(mixedSystem, mm.VerletIntegrator(0.001), platform)
        interpContext = mm.Context(interpSystem, mm.VerletIntegrator(0.001), platform)
        mmContext.setPositions(inpcrd.positions)
        mixedContext.setPositions(inpcrd.positions)
        interpContext.setPositions(inpcrd.positions)
        mmEnergy = mmContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        mixedEnergy = mixedContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpEnergy1 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpContext.setParameter('lambda_interpolate', 0)
        interpEnergy2 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert np.isclose(mixedEnergy, interpEnergy1, rtol=1e-5)
        assert np.isclose(mmEnergy, interpEnergy2, rtol=1e-5)
