import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
import numpy as np
import pytest
rtol = 1e-5

@pytest.mark.parametrize("implementation,platform_name", [
        ('torchani', 'Reference'),
        #('torchani', 'OpenCL'),
        #('torchani', 'CPU'),
        ('nnpops', 'CUDA'),
        #('torchani', 'CUDA')
        ]) 
class TestMLPotential:
    """
    test the `MLPotential.py` functionality, specifically the `createMixedSystem` with interpolation to assert that the energies of interpolated
    systems are consistent with non-interpolated systems using various implementation methods.
    """
    def testCreateMixedSystem(self, implementation, platform_name):
        pdb = app.PDBFile('alanine-dipeptide-explicit.pdb')
        ff = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        mmSystem = ff.createSystem(pdb.topology, nonbondedMethod=app.PME)
        potential = MLPotential('ani2x')
        mlAtoms = [a.index for a in next(pdb.topology.chains()).atoms()]
        mixedSystem = potential.createMixedSystem(pdb.topology, mmSystem, mlAtoms, interpolate=False, implementation = implementation)
        interpSystem = potential.createMixedSystem(pdb.topology, mmSystem, mlAtoms, interpolate=True, implementation = implementation)
        platform = mm.Platform.getPlatformByName(platform_name)
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), platform)
        mixedContext = mm.Context(mixedSystem, mm.VerletIntegrator(0.001), platform)
        interpContext = mm.Context(interpSystem, mm.VerletIntegrator(0.001), platform)
        mmContext.setPositions(pdb.positions)
        mixedContext.setPositions(pdb.positions)
        interpContext.setPositions(pdb.positions)
        mmEnergy = mmContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        mixedEnergy = mixedContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpEnergy1 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpContext.setParameter('lambda', 0)
        interpEnergy2 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert np.isclose(mixedEnergy, interpEnergy1, rtol=rtol), f"mixedEnergy ({mixedEnergy}) does not match interpEnergy1 ({interpEnergy1})"
        assert np.isclose(mmEnergy, interpEnergy2, rtol=rtol), f"mmEnergy ({mmEnergy} does not match interpEnergy2) ({interpEnergy2})"


#if __name__ == '__main__':
#    unittest.main()

