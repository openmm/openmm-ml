import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml import MLPotential

mace = pytest.importorskip("mace", reason="MACE is not installed")
ase = pytest.importorskip("ase", reason="ASE is not installed")
platform_ints = range(mm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestASEPotential:
    def testCalculator(self, platform_int):
        from mace.calculators.foundations_models import mace_off
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential('ase')
        calculator = mace_off('small', default_dtype='float32')
        system = potential.createSystem(pdb.topology, calculator=calculator)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        state = context.getState(energy=True, forces=True)
        energyML = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forceML = state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
        assert np.isclose(energyML, -713468.6327560507, rtol=1e-6)
        assert np.allclose(forceML, (392.8416697946288, -1080.40146230381, 801.0889295457797), rtol=1e-4)

    def testAtoms(self, platform_int):
        from mace.calculators.foundations_models import mace_off
        import ase.io
        path = os.path.join(test_data_dir, "toluene", "toluene.pdb")
        pdb = app.PDBFile(path)
        atoms = ase.io.read(path)
        atoms.calc = mace_off('small', default_dtype='float32')
        potential = MLPotential('ase')
        system = potential.createSystem(pdb.topology, aseAtoms=atoms)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        state = context.getState(energy=True, forces=True)
        energyML = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forceML = state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
        assert np.isclose(energyML, -713468.6327560507, rtol=1e-6)
        assert np.allclose(forceML, (392.8416697946288, -1080.40146230381, 801.0889295457797), rtol=1e-4)

    def testPeriodicSystem(self, platform_int):
        from mace.calculators.foundations_models import mace_off
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        potential = MLPotential('ase')
        calculator = mace_off('small', default_dtype='float32')
        system = potential.createSystem(pdb.topology, calculator=calculator)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        positionsOriginal = pdb.getPositions(asNumpy=True)
        energyRef = -151723354.26015 # Calculated with MACECalculator
        forceRef = (64.17566706460377, 311.2795919353534, 4.956503694474798)
        for i in range(3):
            positions = positionsOriginal + i * 0.9 * unit.nanometers # translate molecule to test PBC
            context.setPositions(positions)
            state = context.getState(energy=True, forces=True)
            energyML = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            forceML = state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
            assert np.isclose(energyML, energyRef, rtol=1e-5)
            assert np.allclose(forceML, forceRef, rtol=1e-3, atol=2e-2)

    def testCreateMixedSystem(self, platform_int):
        from mace.calculators.foundations_models import mace_off
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))
        mlAtoms = list(range(15))
        mmSystem = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential('ase')
        calculator = mace_off('small', default_dtype='float32')
        mixedSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, interpolate=False, calculator=calculator, mlLongRange=False)
        interpSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, interpolate=True, calculator=calculator, mlLongRange=False)
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
