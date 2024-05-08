from sys import stdout

import openmm
import openmm.app as app
import openmm.unit as unit

from openmmml import MLPotential

"""
Uses a deployed trained NequIP model from the basic example on https://github.com/mir-group/nequip
>>> nequip-train configs/example.yaml
>>> nequip-deploy build --train-dir path/to/training/session/ toluene-deployed.pth
"""

# Load toluene structure
pdb = app.PDBFile("toluene.pdb")

# Create a System with NequIP MLP

# Need to specify the unit conversion factors from the NequIP model units to OpenMM units.
# Distance: model is in Angstrom, OpenMM is in nanometers
A_to_nm = 0.1
# Energy: model is in kcal/mol, OpenMM is in kJ/mol
kcal_to_kJ_per_mol = 4.184

potential = MLPotential(
    "nequip",
    modelPath="toluene-deployed.pth",
    lengthScale=A_to_nm,
    energyScale=kcal_to_kJ_per_mol,
)

system = potential.createSystem(pdb.topology)

# Run langevin dynamics at 300K for 1000 steps
integrator = openmm.LangevinIntegrator(
    300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtosecond
)
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.reporters.append(app.DCDReporter("output.dcd", 100))
simulation.reporters.append(
    app.StateDataReporter(
        stdout, 100, step=True, potentialEnergy=True, temperature=True, speed=True
    )
)

# Minimize the energy
simulation.minimizeEnergy()

# Set the velocities to 300K and run 1000 steps
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
simulation.step(1000)
