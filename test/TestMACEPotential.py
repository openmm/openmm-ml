import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml import MLPotential

mace = pytest.importorskip("mace", reason="mace is not installed")
platform_ints = range(mm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestMACE:

    # Reference energies are calculated with MACECalculator
    refEnergy = {
        ('toluene', 'mace-off23-small'): -713468.6327560507,
        ('toluene', 'mace-off23-medium'): -713468.0563706581,
        ('toluene', 'mace-off23-large'): -713467.7476380612,
        ('toluene', 'mace-off24-medium'): -713467.9394350434,
        ('toluene', 'mace-mpa-0-medium'): -8839.299589829867,
        ('toluene', 'mace-omat-0-small'): -8726.63865431241,
        ('toluene', 'mace-omat-0-medium'): -8679.026847088873,
        ('toluene', 'mace-omol-0-extra-large'): -712903.4934289698,
        ('toluene', 'mace-les-off-small'): -713467.9354591698,
        ('toluene', 'mace-polar-1-small'): -712903.1710073923,
        ('toluene', 'mace-polar-1-medium'): -712903.4536792638,
        ('toluene', 'mace-polar-1-large'): -712903.7834631138,
        ('water', 'mace-off23-small'): -43380916.59098946,
        ('water', 'mace-off23-medium'): -43380967.434479,
        ('water', 'mace-off24-medium'): -43380781.370446146,
        ('water', 'mace-mpa-0-medium'): -304547.4706910844,
        ('water', 'mace-omat-0-small'): -303539.62768940086,
        ('water', 'mace-omat-0-medium'): -304131.2834723455,
        ('water', 'mace-les-off-small'): -43381222.49001855,
        ('water', 'mace-polar-1-small'): -43355279.95728109,
        ('water', 'mace-polar-1-medium'): -43355301.06344749,
        ('alanine-dipeptide', 'mace-off23-small'): -151723354.26015,
    }

    @pytest.mark.parametrize("model", ['mace-off23-small', 'mace-off23-medium', 'mace-off23-large', 'mace-off24-medium',
                                       'mace-mpa-0-medium', 'mace-omat-0-small', 'mace-omat-0-medium', 'mace-omol-0-extra-large',
                                       'mace-les-off-small', 'mace-polar-1-small', 'mace-polar-1-medium', 'mace-polar-1-large'])
    def testCreatePureMLSystem(self, platform_int, model):
        if 'mace-les' in model:
            pytest.importorskip("les", reason="les is not installed")
        if 'mace-polar' in model:
            pytest.importorskip("graph_longrange", reason="graph_electrostatics is not installed")
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        system = potential.createSystem(pdb.topology, returnEnergyType='energy')
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        energyML = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert np.isclose(self.refEnergy['toluene', model], energyML, rtol=1e-6)

    @pytest.mark.parametrize(["test", "model"], [
        ('water', 'mace-off23-small'),
        ('water', 'mace-off23-medium'),
        ('water', 'mace-off24-medium'),
        ('water', 'mace-mpa-0-medium'),
        ('water', 'mace-omat-0-small'),
        ('water', 'mace-omat-0-medium'),
        ('water', 'mace-les-off-small'),
        ('water', 'mace-polar-1-small'),
        ('water', 'mace-polar-1-medium'),
        ('alanine-dipeptide', 'mace-off23-small'),
    ])
    def testPeriodicSystem(self, platform_int, test, model):
        if test == 'water':
            pdb = app.PDBFile(os.path.join(test_data_dir, "water", "water.pdb"))
        else:
            pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        potential = MLPotential(model)
        system = potential.createSystem(pdb.topology, returnEnergyType='energy')
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        positionsOriginal = pdb.getPositions(asNumpy=True)
        energyRef = self.refEnergy[test, model]
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
        potential = MLPotential("mace-off23-small")
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

    @pytest.mark.parametrize("precision", ["single", "double"])
    def testPrecisionApplied(self, platform_int, precision):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential('mace-off23-small')

        # Specifying precision single/double.
        system = potential.createSystem(pdb.topology, returnEnergyType='energy', precision=precision)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.positions)

        # Inconsistent dtypes will crash the simulation.
        energyML = context.getState(energy=True, forces=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        # The energy should be physically meaningful under both precisions
        assert np.isfinite(energyML), \
            "Energy is not finite under precision {}".format(precision)
        assert np.isclose(energyML, self.refEnergy['toluene', 'mace-off23-small'], rtol=1e-6),\
            "Energy is not close to reference under precision {}".format(precision)
