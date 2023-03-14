import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
from sys import stdout
from openmmtools.testsystems import AlanineDipeptideVacuum

# Get the system of alanine dipeptide
ala2 = AlanineDipeptideVacuum(constraints=None)

# solvate
ff=openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

modeller=openmm.app.Modeller(ala2.topology, ala2.positions)
modeller.addSolvent(ff, padding=1.0*unit.nanometers)

print(modeller.topology)

mm_system = ff.createSystem(modeller.topology, nonbondedMethod=openmm.app.PME, nonbondedCutoff=1.0*unit.nanometer, constraints=None)

chains = list(modeller.topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]

potential = MLPotential('mace', model_path='MACE_SPICE_larger.model')


mm_ml_system = potential.createMixedSystem(modeller.topology, mm_system, ml_atoms)

# run langevin dynamics at 300K for 10000 steps
integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 1.0*unit.femtosecond)
simulation=app.Simulation(modeller.topology, mm_ml_system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.reporters.append(app.PDBReporter('output.pdb', 100, enforcePeriodicBox=False))
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True,
        potentialEnergy=True, temperature=True, speed=True))


simulation.step(10000)
