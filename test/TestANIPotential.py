import itertools
import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml import MLPotential

torchani = pytest.importorskip("torchani", reason="torchani is not installed")

platform_ints = range(mm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestANIPotential:
    def testSimulate(self, platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, 'toluene', 'toluene.pdb'))
        potential = MLPotential('ani2x')
        system = potential.createSystem(pdb.topology)
        platform = mm.Platform.getPlatform(platform_int)
        integrator = mm.LangevinIntegrator(300.0, 1.0,0.001)
        context = mm.Context(system, integrator, platform)
        context.setPositions(pdb.positions)
        integrator.step(10)
        positions = context.getState(positions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        assert np.all(np.isfinite(positions))

    def testCreateMixedSystem(self, platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, 'alanine-dipeptide', 'alanine-dipeptide-explicit.pdb'))
        ff = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        mmSystem = ff.createSystem(pdb.topology, nonbondedMethod=app.PME)
        potential = MLPotential('ani2x')
        mlAtoms = [a.index for a in next(pdb.topology.chains()).atoms()]
        mixedSystem = potential.createMixedSystem(pdb.topology, mmSystem, mlAtoms, interpolate=False)
        interpSystem = potential.createMixedSystem(pdb.topology, mmSystem, mlAtoms, interpolate=True)
        platform = mm.Platform.getPlatform(platform_int)
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), platform)
        mixedContext = mm.Context(mixedSystem, mm.VerletIntegrator(0.001), platform)
        interpContext = mm.Context(interpSystem, mm.VerletIntegrator(0.001), platform)
        mmContext.setPositions(pdb.positions)
        mixedContext.setPositions(pdb.positions)
        interpContext.setPositions(pdb.positions)
        mmState = mmContext.getState(energy=True, forces=True)
        mixedState = mixedContext.getState(energy=True, forces=True)
        interpState1 = interpContext.getState(energy=True, forces=True)
        interpContext.setParameter('lambda_interpolate', 0)
        interpState2 = interpContext.getState(energy=True, forces=True)
        assert np.isclose(mixedState.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole), interpState1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole), rtol=1e-5)
        assert np.isclose(mmState.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole), interpState2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole), rtol=1e-5)
        assert np.allclose(mixedState.getForces().value_in_unit(unit.kilojoules_per_mole/unit.nanometer), interpState1.getForces().value_in_unit(unit.kilojoules_per_mole/unit.nanometer), rtol=1e-3)
        assert np.allclose(mmState.getForces().value_in_unit(unit.kilojoules_per_mole/unit.nanometer), interpState2.getForces().value_in_unit(unit.kilojoules_per_mole/unit.nanometer), rtol=1e-3)
