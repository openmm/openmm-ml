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
@pytest.mark.parametrize("model", ['mace-off23-small', 'mace-off23-medium', 'mace-off23-large', 'mace-off24-medium',
                                   'mace-mpa-0-medium', 'mace-omat-0-small', 'mace-omat-0-medium'])#, 'mace-omol-0-extra-large'])
class TestMACE:
    def testCreatePureMLSystem(self, platform_int, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        system = potential.createSystem(pdb.topology, returnEnergyType='energy')
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        energyML = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        # Reference energies are calculated with MACECalculator
        refEnergy = {'mace-off23-small': -713468.6327560507,
                     'mace-off23-medium': -713468.0563706581,
                     'mace-off23-large': -713467.7476380612,
                     'mace-off24-medium': -713467.9394350434,
                     'mace-mpa-0-medium': -8839.299589829867,
                     'mace-omat-0-small': -8726.63865431241,
                     'mace-omat-0-medium': -8679.026847088873,
                     'mace-omol-0-extra-large': -712903.4934289698}
        assert np.isclose(refEnergy[model], energyML, rtol=1e-6)
