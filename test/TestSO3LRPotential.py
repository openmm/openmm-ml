import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml import MLPotential

pytest.importorskip("so3lr", reason="so3lr is not installed")
platform_ints = range(mm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Reference energies and forces are calculated with so3lr in double precision (data/so3lr_energies.py).
REFERENCES = {
    'so3lr-tauri-m-2606': (-7873.8672832249595, 248.36537927519996, -1077.7660330814253, 785.2183161574535),
    'charged': (-3202.729232002077, 326.78326157945185, 650.5457688404122, -801.1777044735466),
    'periodic': (-853228.9899654081, 106.03627442943781, 227.65926584593365, 22.6023812377308),
    'periodic-pme': (-873140.6636799071, 104.14584384342513, 246.24814296740152, 24.00634661688487),
}

def createContext(system, platform_int):
    return mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatform(platform_int))

def energyAndForce(context, positions):
    context.setPositions(positions)
    state = context.getState(energy=True, forces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    force = state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
    return energy, force

def assertMatches(energy, force, ref, forceRtol=5e-5, forceAtol=1e-2):
    assert np.isclose(energy, ref[0], rtol=1e-6)
    assert np.allclose(force, ref[1:], rtol=forceRtol, atol=forceAtol)

@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestSO3LR:
    def testCreatePureMLSystem(self, platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        system = MLPotential("so3lr-tauri-m-2606").createSystem(pdb.topology, precision="double")
        energy, force = energyAndForce(createContext(system, platform_int), pdb.getPositions(asNumpy=True))
        assertMatches(energy, force, REFERENCES["so3lr-tauri-m-2606"])

    def testChargedSystem(self, platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, "methanol-ions", "methanol-ions.pdb"))
        system = MLPotential("so3lr-tauri-m-2606").createSystem(pdb.topology, charge=1, precision="double")
        energy, force = energyAndForce(createContext(system, platform_int), pdb.getPositions(asNumpy=True))
        assertMatches(energy, force, REFERENCES["charged"])

    @pytest.mark.parametrize("args, refKey", [
        (dict(kspace="none"), "periodic"),
        ({}, "periodic-pme"),
    ])
    def testPeriodicSystem(self, platform_int, args, refKey):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        system = MLPotential("so3lr-tauri-m-2606", **args).createSystem(pdb.topology, precision="double")
        context = createContext(system, platform_int)
        positionsOriginal = pdb.getPositions(asNumpy=True)
        for i in range(3):
            # Translate the molecule to test PBC.
            energy, force = energyAndForce(context, positionsOriginal + i * 0.9 * unit.nanometers)
            assertMatches(energy, force, REFERENCES[refKey], forceRtol=1e-3, forceAtol=2e-2)

    def testCreateMixedSystem(self, platform_int):
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))
        mlAtoms = list(range(15))
        mmSystem = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential("so3lr-tauri-m-2606")
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

# These do not depend on the platform.

def testUnsupportedElement():
    # The original model covers H, C, N, O, F, P, S and Cl, not Na.
    pdb = app.PDBFile(os.path.join(test_data_dir, "methanol-ions", "methanol-ions.pdb"))
    with pytest.raises(ValueError, match="unsupported element"):
        MLPotential("so3lr").createSystem(pdb.topology, charge=1)

def testLongRangeFlag():
    # PME by default, except for the original model, which stays in real space.
    assert MLPotential("so3lr-tauri-m-2606")._impl.getMLLongRange()
    assert not MLPotential("so3lr")._impl.getMLLongRange()
    assert MLPotential("so3lr", kspace="pme")._impl.getMLLongRange()
    assert not MLPotential("so3lr-tauri-m-2606", kspace="none")._impl.getMLLongRange()
