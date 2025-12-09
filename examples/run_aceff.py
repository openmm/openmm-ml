import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
import os
from sys import stdout


pdb = app.PDBFile("../test/data/toluene/toluene.pdb")
potential = MLPotential("aceff-1.1") 
system = potential.createSystem(pdb.topology, charge=0)

integrator = mm.LangevinIntegrator(
    300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtosecond
)
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.reporters.append(app.PDBReporter("output.pdb", 100))
simulation.reporters.append(
    app.StateDataReporter(
        stdout, 100, step=True, potentialEnergy=True, temperature=True, speed=True
    )
)

simulation.minimizeEnergy()
simulation.step(1000)
     
