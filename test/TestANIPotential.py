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

    def testForces(self, platform_int):
        # The forces must be the gradient of the energy, in kJ/mol/nm.
        pdb = app.PDBFile(os.path.join(test_data_dir, 'toluene', 'toluene.pdb'))
        potential = MLPotential('ani2x')
        system = potential.createSystem(pdb.topology)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.positions)
        forces = context.getState(forces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
        import torch
        model = torchani.models.ANI2x(periodic_table_index=True).double()
        species = torch.tensor([[atom.element.atomic_number for atom in pdb.topology.atoms()]])
        positions = torch.tensor(pdb.positions.value_in_unit(unit.angstrom), dtype=torch.float64).unsqueeze(0)
        positions.requires_grad_(True)
        model((species, positions)).energies.backward()
        expected = (-positions.grad[0]*torchani.units.hartree2kjoulemol(1)*10).numpy()
        assert np.allclose(forces, expected, rtol=1e-3, atol=0.1)

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
        assert np.allclose(mixedState.getForces().value_in_unit(unit.kilojoules_per_mole/unit.nanometer), interpState1.getForces().value_in_unit(unit.kilojoules_per_mole/unit.nanometer), rtol=1e-3, atol=1e-4)
        assert np.allclose(mmState.getForces().value_in_unit(unit.kilojoules_per_mole/unit.nanometer), interpState2.getForces().value_in_unit(unit.kilojoules_per_mole/unit.nanometer), rtol=1e-3, atol=1e-4)
