# TODO: This script should ideally compute reference energies for EMLE without
# using OpenMM-ML or calling the EMLE model directly.

import emle.models
import numpy as np
import openmm
import openmm.app
import openmm.unit as unit
import openmmml
import torch

pdb = openmm.app.PDBFile("alanine-dipeptide/alanine-dipeptide-explicit.pdb")

chains = list(pdb.topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]
mm_atoms = [atom.index for chain in chains[1:] for atom in chain.atoms()]
atomic_numbers = np.array([atom.element.atomic_number for atom in chains[0].atoms()], dtype=int)

# Make a mixed system with OpenMM-ML using mechanical embedding.

mm_system = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml").createSystem(pdb.topology, nonbondedMethod=openmm.app.PME)

for i_force, force in enumerate(mm_system.getForces()):
    if isinstance(force, openmm.NonbondedForce):
        charges_mm = np.array([force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in mm_atoms])
        charge_ml = round(sum(force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in ml_atoms))

ml_system = openmmml.MLPotential("mace-off23-small").createMixedSystem(pdb.topology, mm_system, ml_atoms)

# Zero out the ML charges (as is necessary when using SIRE).

for i_force, force in enumerate(ml_system.getForces()):
    if isinstance(force, openmm.NonbondedForce):
        for i in ml_atoms:
            _, sigma, epsilon = force.getParticleParameters(i)
            force.setParticleParameters(i, 0, sigma, epsilon)

context = openmm.Context(ml_system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatform("Reference"))
context.setPositions(pdb.positions)
context_energy = context.getState(energy=True).getPotentialEnergy()

# Get the EMLE energy.

xyz = np.array(pdb.positions.value_in_unit(unit.angstrom))
emle = emle.models.EMLE(cutoff=7.5, dtype=torch.float64, device=torch.device("cpu"))
emle_energy = emle(
    torch.tensor(atomic_numbers),
    torch.tensor(charges_mm, dtype=torch.float64),
    torch.tensor(xyz[ml_atoms], dtype=torch.float64),
    torch.tensor(xyz[mm_atoms], dtype=torch.float64),
    torch.tensor(pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.angstrom), dtype=torch.float64),
    charge_ml,
    preprocess=True,
    use_switching_function=True
).sum().item() * unit.hartree / unit.item

print((context_energy + emle_energy).value_in_unit(unit.kilojoule_per_mole))
