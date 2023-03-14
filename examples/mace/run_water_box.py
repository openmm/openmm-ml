import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
from sys import stdout


pdb=app.PDBFile("water.pdb")

potential = MLPotential('mace', model_path='MACE_SPICE_larger.model')


system = potential.createSystem(pdb.topology)

integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 1.0*unit.femtosecond)
simulation=app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.reporters.append(app.PDBReporter('output.pdb', 100))
simulation.reporters.append(app.StateDataReporter(stdout, 10, step=True,
        potentialEnergy=True, temperature=True, speed=True))


simulation.step(100)
