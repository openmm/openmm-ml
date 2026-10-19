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

    # Reference energies and forces are calculated with MACECalculator
    refEnergyForce = {
        ('toluene', 'mace-off23-small'): (-713468.6327560507, 392.8416697946288, -1080.40146230381, 801.0889295457797),
        ('toluene', 'mace-off23-medium'): (-713468.0563706581, 374.8117102505617, -1079.01407338965, 801.9595104181037),
        ('toluene', 'mace-off23-large'): (-713467.7476380612, 380.15388510527805, -1090.5446637960054, 802.1902663923878),
        ('toluene', 'mace-off24-medium'): (-713467.9394350434, 383.0395851333961, -1080.7707725638033, 803.3884045206088),
        ('toluene', 'mace-mpa-0-medium'): (-8839.299589829867, 0.23499198257923126, -978.909423828125, 566.8569946289062),
        ('toluene', 'mace-omat-0-small'): (-8726.63865431241, 18.13302230834961, -852.846923828125, 443.0360107421875),
        ('toluene', 'mace-omat-0-medium'): (-8679.026847088873, 128.75936889648438, -901.26416015625, 564.5759887695312),
        ('toluene', 'mace-omol-0-extra-large'): (-712903.4934289698, 318.45394974859335, -1087.9585392893366, 792.5752993645389),
        ('toluene', 'mace-les-off-small'): (-713467.9354591698, 377.32413753660404, -1091.193630901418, 806.3824675756106),
        ('toluene', 'mace-polar-1-small'): (-712903.1710073923, 317.0723876953125, -1084.5985107421875, 793.995361328125),
        ('toluene', 'mace-polar-1-medium'): (-712903.4536792638, 318.300537109375, -1087.6939697265625, 792.6447143554688),
        ('toluene', 'mace-polar-1-large'): (-712903.7834631138, 319.8680419921875, -1087.9964599609375, 793.3919677734375),
        ('water', 'mace-off23-small'): (-43380916.59098946, -154.50224890831967, -253.93971262845008, -660.3905756253831),
        ('water', 'mace-off23-medium'): (-43380967.434479, -146.3675641299843, -273.3302945743585, -731.808179128405),
        ('water', 'mace-off24-medium'): (-43380781.370446146, -250.48114288876602, -269.1192191922664, -723.1723444623115),
        ('water', 'mace-mpa-0-medium'): (-304547.517803063, -173.9760284423828, 177.88882446289062, -62.7952880859375),
        ('water', 'mace-omat-0-small'): (-303539.7690253366, -119.0258560180664, 193.8815460205078, -69.38377380371094),
        ('water', 'mace-omat-0-medium'): (-304131.3305843241, -105.28791809082031, 181.89874267578125, -91.41193389892578),
        ('water', 'mace-les-off-small'): (-43381222.49001855, -138.59063239924518, -282.4272721140492, -739.4014711910415),
        ('water', 'mace-polar-1-small'): (-43355282.972447716, -171.88583374023438, -140.40577697753906, -530.1849975585938),
        ('water', 'mace-polar-1-medium'): (-43355301.06344749, -184.02772521972656, -158.17813110351562, -576.86962890625),
        ('alanine-dipeptide', 'mace-off23-small'): (-151723354.26015, 64.17566706460377, 311.2795919353534, 4.956503694474798),
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
        state = context.getState(energy=True, forces=True)
        energyML = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forceML = state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
        assert np.isclose(energyML, self.refEnergyForce['toluene', model][0], rtol=1e-6)
        assert np.allclose(forceML, self.refEnergyForce['toluene', model][1:], rtol=1e-3, atol=2e-2)

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
        energyRef, *forceRef = self.refEnergyForce[test, model]
        for i in range(3):
            positions = positionsOriginal + i * 0.9 * unit.nanometers # translate molecule to test PBC
            context.setPositions(positions)
            state = context.getState(energy=True, forces=True)
            energyML = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            forceML = state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
            assert np.isclose(energyML, energyRef, rtol=1e-5)
            assert np.allclose(forceML, forceRef, rtol=1e-3, atol=2e-2)

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
        assert np.isclose(energyML, self.refEnergyForce['toluene', 'mace-off23-small'][0], rtol=1e-6),\
            "Energy is not close to reference under precision {}".format(precision)
