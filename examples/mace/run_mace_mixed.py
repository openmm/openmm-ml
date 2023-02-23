import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
from sys import stdout
from openmmtools.testsystems import AlanineDipeptideVacuum
import time
import torch.autograd.profiler as profiler

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


mm_ml_system = potential.createMixedSystem(modeller.topology, mm_system, ml_atoms,  dtype="float64")

# run langevin dynamics at 300K for 1000 steps
integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 1.0*unit.femtosecond)
simulation=app.Simulation(modeller.topology, mm_ml_system, integrator, platform=openmm.Platform.getPlatformByName("CPU"))
simulation.context.setPositions(modeller.positions)
#simulation.reporters.append(app.PDBReporter('output.pdb', 10, enforcePeriodicBox=False))
simulation.reporters.append(app.StateDataReporter(stdout, 10, step=True,
        potentialEnergy=True, temperature=True, speed=True))

# warm up
simulation.step(100)

# time
Nstep = 100
t1 = time.perf_counter()
simulation.step(Nstep)
t2 = time.perf_counter()

print("time for 100 steps:", t2-t1, "s")
print("time for one steps:", ((t2-t1)/Nstep)*1000.0, "ms")


# profile
with profiler.profile() as prof:
  simulation.step(1)

print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=20))

# Minimize the energy
# simulation.minimizeEnergy()
# energy=simulation.context.getState(getEnergy=True).getPotentialEnergy()
# print(energy, energy.in_units_of(unit.kilocalorie_per_mole))
