# TODO: This script should ideally compute reference energies for EMLE without
# using OpenMM-ML or calling the EMLE model directly, but this is not the case
# as the documented way to invoke EMLE with SIRE is not usable.

import emle.calculator
import numpy as np
import openmm
import openmm.app
import openmm.unit as unit
import openmmml

pdb = openmm.app.PDBFile("alanine-dipeptide/alanine-dipeptide-explicit.pdb")

chains = list(pdb.topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]
mm_atoms = [atom.index for chain in chains[1:] for atom in chain.atoms()]
atomic_numbers = np.array([atom.element.atomic_number for atom in chains[0].atoms()], dtype=int)

for periodic in (False, True):
    # Make a mixed system with OpenMM-ML using mechanical embedding.

    mm_system = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml").createSystem(pdb.topology, nonbondedMethod=openmm.app.PME if periodic else openmm.app.NoCutoff)

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

    # Get the EMLE energy by calling EMLECalculator.

    xyz = np.array(pdb.positions.value_in_unit(unit.angstrom))
    emle_calculator = emle.calculator.EMLECalculator(backend=None, device="cpu")
    emle_energy = float(emle_calculator._calculate_energy_and_gradients(
        atomic_numbers,
        charges_mm,
        xyz[ml_atoms],
        xyz[mm_atoms],
        cell=np.array(pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.angstrom)) if periodic else None,
        charge=charge_ml
    )[2]) * unit.hartree / unit.item

    print(periodic, (context_energy + emle_energy).value_in_unit(unit.kilojoule_per_mole))
