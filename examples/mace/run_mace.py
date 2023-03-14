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


integrator = openmm.LangevinIntegrator(300*unit.kelvin, 10.0/unit.picoseconds, 1.0*unit.femtosecond)
simulation=app.Simulation(ala2.topology, system, integrator)
simulation.context.setPositions(ala2.positions)
simulation.reporters.append(app.PDBReporter('output.pdb', 100))
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True,
        potentialEnergy=True, temperature=True, speed=True))

simulation.step(10000)
