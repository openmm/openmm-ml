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
    def _toluene_energy(self, name, platform_int, modelPath=None, **createSystemArgs):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potentialArgs = {} if modelPath is None else {"modelPath": modelPath}
        system = MLPotential(name, **potentialArgs).createSystem(pdb.topology, **createSystemArgs)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    # Reference toluene energies (kJ/mol) for the pretrained families, calculated with
    # AIMNet2ASE (see test/data/toluene/aimnet2_energies.py).  Toluene is neutral and
    # H/C only, so it is compatible with every family, including rxn (net-neutral,
    # H/C/N/O only).  Energy references differ across families (rxn in particular uses a
    # learned, shifted scale), so these values are not comparable to one another.
    familyEnergies = {
        "aimnet2": -713468.0026230365,
        "aimnet2-2025": -712625.9531791345,
        "aimnet2-nse": -713470.4078623096,
        "aimnet2-pd": -712635.0374879715,
        "aimnet2-rxn": -208.3835827516966,
    }

    @pytest.mark.parametrize("model,energyRef", list(familyEnergies.items()))
    def testCreateModels(self, platform_int, model, energyRef):
        # Each pretrained family reproduces the reference energy computed
        # independently with AIMNet2ASE.
        assert np.isclose(self._toluene_energy(model, platform_int), energyRef, rtol=rtol)

    def testLocalModel(self, platform_int):
        # Loading the pretrained aimnet2 model as a custom model via the 'aimnet' name
        # and an explicit modelPath gives the same energy as loading it by the 'aimnet2'
        # name (tested in testCreateModels).
        from aimnet.calculators.calculator import get_model_path
        energy = self._toluene_energy("aimnet", platform_int, modelPath=get_model_path("aimnet2"))
        assert np.isclose(energy, self.familyEnergies["aimnet2"], rtol=rtol)

    def testModelIndex(self, platform_int):
        # The family name selects ensemble member 0, so modelIndex=0 reproduces the
        # family default energy while a different member gives a slightly different energy.
        # Check they're relatively similar but do actually differ.
        default = self._toluene_energy("aimnet2", platform_int)
        member0 = self._toluene_energy("aimnet2", platform_int, modelIndex=0)
        member2 = self._toluene_energy("aimnet2", platform_int, modelIndex=2)
        assert np.isclose(default, member0, rtol=rtol)
        assert abs(default - member2) > 0.1

    def testInvalidModelIndex(self, platform_int):
        # An out-of-range member and modelIndex on a local model both raise ValueError.
        from aimnet.calculators.calculator import get_model_path
        with pytest.raises(ValueError):
            self._toluene_energy("aimnet2", platform_int, modelIndex=4)
        with pytest.raises(ValueError):
            self._toluene_energy("aimnet", platform_int, modelPath=get_model_path("aimnet2"), modelIndex=0)

    def testPeriodicSystem(self, platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        potential = MLPotential("aimnet2", charge=0, multiplicity=1)
        system = potential.createSystem(pdb.topology)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        positionsOriginal = pdb.getPositions(asNumpy=True)
        energyRef = -151715123.01342362 # in kJ/mol, calculated using the AIMNet2ASE
        for i in range(3):
            positions = positionsOriginal + i * 0.9 * unit.nanometers # translate molecule to test PBC
            context.setPositions(positions)
            energyML = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            assert np.isclose(energyRef, energyML, rtol=rtol)

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
