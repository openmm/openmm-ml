import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
from sys import stdout



"""
Uses a deployed trained NequiP model, toy model with PBC:
nequip-train configs/minimal_toy_emt.yaml
nequip-deploy build --train-dir path/to/training/session/ deployed_model.pth
"""

### build the inital structure for the toy model
import ase.build
import ase.io
import numpy as np

rng = np.random.default_rng(123)
supercell = (4, 4, 4)
sigma = 0.1
base_atoms = ase.build.bulk('Cu', 'fcc', cubic=True).repeat(supercell)
print(base_atoms)
topology=openmm.app.Topology()
element = app.Element.getBySymbol('Cu')
chain = topology.addChain()
for particle in range(len(base_atoms)):
        residue = topology.addResidue('Cu', chain)
        topology.addAtom('Cu', element, residue)
cell=np.array(base_atoms.cell[:])*0.1 # A -> nm
topology.setPeriodicBoxVectors(cell)
print(topology)
positions=base_atoms.positions*0.1 # A->nm


# create a System with NequIP MLP

# need to specify the unit conversion factors from the NequIP model units to OpenMM units.
# distance: model is in Angstrom, OpenMM is in nanometers
A_to_nm = 0.1
# energy: model is in eV, OpenMM is in kJ/mol
eV_to_kJ_per_mol = 96.49

potential = MLPotential('nequip', model_path='minimal_toy_emt_model_deployed.pth', 
                        distance_to_nm=A_to_nm, 
                        energy_to_kJ_per_mol=eV_to_kJ_per_mol)

system = potential.createSystem(topology)


# run NPT
temperature=800.0*unit.kelvin
pressure=1.0*unit.bar
integrator = openmm.LangevinIntegrator(temperature, 1.0/unit.picoseconds, 5.0*unit.femtosecond)
simulation=app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)

system.addForce(openmm.MonteCarloBarostat(pressure, temperature))
simulation.context.reinitialize(preserveState=True)

simulation.reporters.append(app.PDBReporter('output.pdb', 100, enforcePeriodicBox=True))
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True,
        potentialEnergy=True, temperature=True, volume=True))

simulation.step(1000)

# Minimize the energy
simulation.minimizeEnergy()
energy=simulation.context.getState(getEnergy=True).getPotentialEnergy()
print(energy)
