import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
from sys import stdout
from openmmtools.testsystems import AlanineDipeptideVacuum

# Get the system of alanine dipeptide
ala2 = AlanineDipeptideVacuum(constraints=None)

# Remove MM forces
while ala2.system.getNumForces() > 0:
  ala2.system.removeForce(0)

# The system should not contain any additional force and constrains
assert ala2.system.getNumConstraints() == 0
assert ala2.system.getNumForces() == 0

potential = MLPotential('mace', model_path='MACE_SPICE_larger.model')


system = potential.createSystem(ala2.topology)

# run langevin dynamics at 300K for 1000 steps
integrator = openmm.LangevinIntegrator(300*unit.kelvin, 10.0/unit.picoseconds, 1.0*unit.femtosecond)
simulation=app.Simulation(ala2.topology, system, integrator, platform=openmm.Platform.getPlatformByName("CPU"))
simulation.context.setPositions(ala2.positions)
simulation.reporters.append(app.PDBReporter('output.pdb', 10))
simulation.reporters.append(app.StateDataReporter(stdout, 10, step=True,
        potentialEnergy=True, temperature=True, speed=True))

simulation.step(1000)

# Minimize the energy
simulation.minimizeEnergy()
energy=simulation.context.getState(getEnergy=True).getPotentialEnergy()
print(energy, energy.in_units_of(unit.kilocalorie_per_mole))
