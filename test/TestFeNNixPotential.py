import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml import MLPotential

fennol = pytest.importorskip("fennol", reason="FeNNol is not installed")
platform_ints = range(mm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestFeNNix:
    @pytest.mark.parametrize("model", ["fennix-bio1-small", "fennix-bio1-medium"])
    def testCreatePureMLSystem(self, platform_int, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        system = potential.createSystem(pdb.topology, precision="double")
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        state = context.getState(energy=True, forces=True)
        energyML = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forceML = state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
        # Reference energies are calculated with FENNIXCalculator
        refEnergyForce = {
            "fennix-bio1-small": (-5.200859421605385, 299.72070579954755, -1076.031725516552, 809.5794246552945),
            "fennix-bio1-medium": (-2.302869698698856, 295.7830852973222, -1083.4857057173424, 812.9159403689472),
        }
        assert np.isclose(refEnergyForce[model][0], energyML, rtol=1e-5)
        assert np.allclose(refEnergyForce[model][1:], forceML, rtol=1e-5)

    @pytest.mark.parametrize("model", ["fennix-bio1-small", "fennix-bio1-medium", "fennix-bio1-small-finetune-ions", "fennix-bio1-medium-finetune-ions"])
    def testChargedSystem(self, platform_int, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "methanol-ions", "methanol-ions.pdb"))
        potential = MLPotential(model)
        system = potential.createSystem(pdb.topology, charge=1, precision="double")
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        state = context.getState(energy=True, forces=True)
        energyML = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forceML = state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
        # Reference energies are calculated with FENNIXCalculator
        refEnergyForce = {
            "fennix-bio1-small": (-599.6015619222411, 86.6464241697355, 593.573689946066, -628.1270324646932),
            "fennix-bio1-medium": (-1109.208807488106, 373.11371113240824, 743.1925118675491, -906.6656704330803),
            "fennix-bio1-small-finetune-ions": (-560.4959154537403, 191.59096961826992, 591.9506414086745, -684.621698012006),
            "fennix-bio1-medium-finetune-ions": (-1068.531665542107, 350.80378391292976, 743.0344742389011, -897.2904849616576),
        }
        assert np.isclose(refEnergyForce[model][0], energyML, rtol=1e-5)
        assert np.allclose(refEnergyForce[model][1:], forceML, rtol=1e-5)

    def testPeriodicSystem(self, platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        potential = MLPotential("fennix-bio1-small")
        system = potential.createSystem(pdb.topology, precision="double")
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        positionsOriginal = pdb.getPositions(asNumpy=True)
        energyRef = -68462.99055925063 # Calculated with FENNIXCalculator
        forceRef = (47.30085923811878, 375.1738740617789, -5.698304842660946)
        for i in range(3):
            positions = positionsOriginal + i * 0.9 * unit.nanometers # translate molecule to test PBC
            context.setPositions(positions)
            state = context.getState(energy=True, forces=True)
            energyML = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            forceML = state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
            assert np.isclose(energyML, energyRef, rtol=1e-5)
            assert np.allclose(forceML, forceRef, rtol=1e-3, atol=1e-2)

    def testCreateMixedSystem(self, platform_int):
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))
        mlAtoms = list(range(15))
        mmSystem = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential("fennix-bio1-small")
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
