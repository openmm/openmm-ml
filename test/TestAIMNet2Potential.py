import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml import MLPotential

aimnet = pytest.importorskip("aimnet", reason="aimnet2 is not installed")
rtol = 1e-5
platform_ints = range(mm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestAIMNet2:
    def _toluene_energy_force(self, name, platform_int, modelPath=None, **createSystemArgs):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potentialArgs = {} if modelPath is None else {"modelPath": modelPath}
        system = MLPotential(name, **potentialArgs).createSystem(pdb.topology, **createSystemArgs)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        state = context.getState(energy=True, forces=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole), state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)

    # Reference toluene energies (kJ/mol) and forces (kJ/mol/nm) for the pretrained families,
    # calculated with AIMNet2ASE (see test/data/aimnet2_energies.py).  Toluene is neutral and
    # H/C only, so it is compatible with every family, including rxn (net-neutral,
    # H/C/N/O only).  Energy references differ across families (rxn in particular uses a
    # learned, shifted scale), so these values are not comparable to one another.
    familyEnergiesForces = {
        "aimnet2": (-713468.003174878, 421.1900634765625, -1076.18798828125, 785.7400512695312),
        "aimnet2-2025": (-712625.9538447304, 91.4974365234375, -1007.3394775390625, 759.072509765625),
        "aimnet2-nse": (-713470.4084614277, 405.39129638671875, -1087.56591796875, 805.9398193359375),
        "aimnet2-pd": (-712635.0375372141, 117.314453125, -996.3536987304688, 761.0477294921875),
        "aimnet2-rxn": (-208.3845615995897, 390.8122863769531, -1098.330322265625, 822.995849609375),
    }

    @pytest.mark.parametrize("model,energyForceRef", list(familyEnergiesForces.items()))
    def testCreateModels(self, platform_int, model, energyForceRef):
        # Each pretrained family reproduces the reference energy computed
        # independently with AIMNet2ASE.
        energyML, forceML = self._toluene_energy_force(model, platform_int)
        assert np.isclose(energyML, energyForceRef[0], rtol=rtol)
        assert np.allclose(forceML, energyForceRef[1:], rtol=rtol)

    def testLocalModel(self, platform_int):
        # Loading the pretrained aimnet2 model as a custom model via the 'aimnet' name
        # and an explicit modelPath gives the same energy as loading it by the 'aimnet2'
        # name (tested in testCreateModels).
        from aimnet.calculators.calculator import get_model_path
        energy, force = self._toluene_energy_force("aimnet", platform_int, modelPath=get_model_path("aimnet2"))
        assert np.isclose(energy, self.familyEnergiesForces["aimnet2"][0], rtol=rtol)
        assert np.allclose(force, self.familyEnergiesForces["aimnet2"][1:], rtol=rtol)

    def testModelIndex(self, platform_int):
        # The family name selects ensemble member 0, so modelIndex=0 reproduces the
        # family default energy while a different member gives a slightly different energy.
        # Check they're relatively similar but do actually differ.
        default = self._toluene_energy_force("aimnet2", platform_int)[0]
        member0 = self._toluene_energy_force("aimnet2", platform_int, modelIndex=0)[0]
        member2 = self._toluene_energy_force("aimnet2", platform_int, modelIndex=2)[0]
        assert np.isclose(default, member0, rtol=rtol)
        assert abs(default - member2) > 0.1

    def testInvalidModelIndex(self, platform_int):
        # An out-of-range member and a non-default modelIndex on a local model both raise
        # ValueError.
        from aimnet.calculators.calculator import get_model_path
        with pytest.raises(ValueError):
            self._toluene_energy_force("aimnet2", platform_int, modelIndex=4)
        with pytest.raises(ValueError):
            self._toluene_energy_force("aimnet", platform_int, modelPath=get_model_path("aimnet2"), modelIndex=2)

    def testUnsupportedModel(self, platform_int):
        # A name that reaches the impl but isn't a known family (e.g. the legacy b973c
        # family, a raw member alias, or a typo) is rejected.  Names are normally gated by
        # the registered entry points before this point, so we exercise the impl directly.
        from openmmml.models.aimnet2potential import AIMNet2PotentialImpl
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        for name in ("aimnet2-b973c", "aimnet2_wb97m_d3_2", "not-a-model"):
            impl = AIMNet2PotentialImpl(name)
            with pytest.raises(ValueError, match="Unsupported AIMNet2 model"):
                impl.addForces(pdb.topology, mm.System(), None, 0)

    def testPeriodicSystem(self, platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        potential = MLPotential("aimnet2", charge=0, multiplicity=1)
        system = potential.createSystem(pdb.topology)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        positionsOriginal = pdb.getPositions(asNumpy=True)
        energyRef = -151715123.01342362 # in kJ/mol, calculated using the AIMNet2ASE
        forceRef = (-79.20823669433594, 296.98980712890625, -55.57414627075195)
        for i in range(3):
            positions = positionsOriginal + i * 0.9 * unit.nanometers # translate molecule to test PBC
            context.setPositions(positions)
            state = context.getState(energy=True, forces=True)
            energyML = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            forceML = state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
            assert np.isclose(energyRef, energyML, rtol=rtol)
            assert np.allclose(forceRef, forceML, rtol=1e-3)

    def testCreateMixedSystem(self, platform_int):
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))
        mlAtoms = list(range(15))
        mmSystem = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential("aimnet2", charge=0, multiplicity=1)
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
        assert np.isclose(mixedEnergy, interpEnergy1, rtol=rtol)
        assert np.isclose(mmEnergy, interpEnergy2, rtol=rtol)
