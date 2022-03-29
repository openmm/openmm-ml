import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
import unittest
rtol = 1e-5

class TestMLPotential(unittest.TestCase):
    """
    test the `MLPotential.py` functionality, specifically the `createMixedSystem` with interpolation to assert that the energies of interpolated
    systems are consistent with non-interpolated systems using various implementation methods.
    """

    @parameterized.expand([
        ['torchani', 'Reference'],
        ['torchani', 'OpenCL'],
        ['torchani', 'CPU'],
        ['nnpops', 'CUDA'],
        ['torchani', 'CUDA']
        ])
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
        self.assertAlmostEqual(mixedEnergy, interpEnergy1, delta=rtol*abs(mixedEnergy))
        self.assertAlmostEqual(mmEnergy, interpEnergy2, delta=rtol*abs(mmEnergy))


if __name__ == '__main__':
    unittest.main()

