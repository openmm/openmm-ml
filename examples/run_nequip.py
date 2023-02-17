import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
from sys import stdout

"""
Uses a deployed trained NequiP model, toluene example:
nequip-train configs/example.yaml
nequip-deploy build --train-dir path/to/training/session/ example_model_deployed.pth
"""

# load toluene structure
pdb = app.PDBFile("toluene.pdb")

# create a System with NequIP MLP

# need to specify the unit conversion factors from the NequIP model units to OpenMM units.
# distance: model is in Angstrom, OpenMM is in nanometers
A_to_nm = 0.1
# energy: model is in kcal/mol, OpenMM is in kJ/mol
kcal_to_kJ_per_mol = 4.184

potential = MLPotential('nequip', model_path='example_model_deployed.pth',
                        distance_to_nm=A_to_nm,
                        energy_to_kJ_per_mol=kcal_to_kJ_per_mol)

system = potential.createSystem(pdb.topology)

# run langevin dynamics at 300K for 1000 steps
integrator = openmm.LangevinIntegrator(300*unit.kelvin, 10.0/unit.picoseconds, 1.0*unit.femtosecond)
simulation=app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.reporters.append(app.PDBReporter('output.pdb', 10))
simulation.reporters.append(app.StateDataReporter(stdout, 10, step=True,
        potentialEnergy=True, temperature=True))

simulation.step(1000)

# Minimize the energy
simulation.minimizeEnergy()
energy=simulation.context.getState(getEnergy=True).getPotentialEnergy()
print(energy, energy.in_units_of(unit.kilocalorie_per_mole))
