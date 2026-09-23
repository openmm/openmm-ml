# This script computes reference energies for EMLE using Sire.

import os

import openmm
import openmm.app
import openmm.unit as unit
import openmmml
import torch
from emle.models import EMLE

import sire as sr

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alanine-dipeptide")
pdb = openmm.app.PDBFile(os.path.join(data_dir, "alanine-dipeptide-explicit.pdb"))

chains = list(pdb.topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]

cutoff = 7.5
switch_width = 0.2
model = EMLE(
    cutoff=cutoff,
    switch_width=switch_width,
    dtype=torch.float32,
    device=torch.device("cpu"),
)

for periodic in (True, False):
    # Make a mixed system with OpenMM-ML using mechanical embedding.

    mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    mm_system = mm_force_field.createSystem(
        pdb.topology,
        nonbondedMethod=openmm.app.PME if periodic else openmm.app.NoCutoff,
    )
    ml_system = openmmml.MLPotential("mace-off23-small").createMixedSystem(
        pdb.topology, mm_system, ml_atoms
    )

    # Zero out the ML charges and compute the ML/MM energy without the electrosttaics.

    for i_force, force in enumerate(ml_system.getForces()):
        if isinstance(force, openmm.NonbondedForce):
            for i in ml_atoms:
                _, sigma, epsilon = force.getParticleParameters(i)
                force.setParticleParameters(i, 0, sigma, epsilon)

    context = openmm.Context(
        ml_system,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatform("Reference"),
    )
    context.setPositions(pdb.positions)
    mechanical_energy = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )

    # Get the EMLE electrostatic embedding energy from a Sire QM/MM engine.

    mols = sr.load(
        os.path.join(data_dir, "alanine-dipeptide-explicit.prmtop"),
        os.path.join(data_dir, "alanine-dipeptide-explicit.inpcrd"),
    )

    qm_mols, engine = sr.qm.emle(
        mols,
        ml_atoms,
        model,
        cutoff=f"{cutoff}A",
        neighbour_list_frequency=0,
        switch_width=switch_width,
    )

    d = qm_mols.dynamics(
        timestep="1fs",
        constraint="none",
        platform="cpu",
        qm_engine=engine,
        lambda_interpolate=1.0,
        vacuum=not periodic,
    )
    sire_context = d._d._omm_mols

    qm_forces = [
        f for f in sire_context.getSystem().getForces() if "QMForce" in f.getName()
    ]
    assert len(qm_forces) == 1, "Could not find the QM force in the OpenMM system"
    qm_force = qm_forces[0]

    state = sire_context.getState(getEnergy=True, groups={qm_force.getForceGroup()})
    embedding_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    label = "periodic" if periodic else "non-periodic"
    print(f"alanine-dipeptide ({label}): {mechanical_energy + embedding_energy}")
