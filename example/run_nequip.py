import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
from sys import stdout

"""
Uses a deployed trained NequiP model, toluene example:
nequip-train configs/example.yaml
nequip-deploy build --train-dir path/to/training/session/ deployed_model.pth
"""

# load toluene structure
pdb = app.PDBFile("toluene.pdb")


# create a System with NequIP MLP
potential = MLPotential('nequip', model_path="deployed_model.pth", atom_type_to_atomic_number={"H":1, "C":6})
system = potential.createSystem(pdb.topology)


# run langevin dynamics at 300K for 1000 steps
integrator = openmm.LangevinIntegrator(300*unit.kelvin, 10.0/unit.picoseconds, 1.0*unit.femtosecond)
simulation=app.Simulation(pdb.topology, system, integrator, platform=openmm.Platform.getPlatformByName("CPU"))
simulation.context.setPositions(pdb.positions)
simulation.reporters.append(app.PDBReporter('output.pdb', 10))
simulation.reporters.append(app.StateDataReporter(stdout, 10, step=True,
        potentialEnergy=True, temperature=True))

simulation.step(1000)

# Minimize the energy
simulation.minimizeEnergy()
energy=simulation.context.getState(getEnergy=True).getPotentialEnergy()
print(energy, energy.in_units_of(unit.kilocalorie_per_mole))