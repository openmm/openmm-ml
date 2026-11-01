# This script computes reference energies for the PolarMACE electrostatic embedding.
#
# The embedded PolarMACE energy is evaluated with MACE and graph_longrange alone: the model's
# electrostatic blocks are wrapped with graph_longrange's external-source wrappers and the MM
# charges are passed to them directly.  OpenMM-ML is used only for the MM part, through its
# generic mechanical embedding with an unrelated ML model whose force is placed in its own
# force group and left out; none of OpenMM-ML's electrostatic-embedding code is used.

import os

import ase
import numpy as np
import openmm
import openmm.app
import openmm.unit as unit
import openmmml
import torch
from graph_longrange.external_source_energy import GTOElectrostaticExternalSourceEnergy
from graph_longrange.external_source_features import GTOElectrostaticExternalSourceFeatures
from mace.calculators.foundations_models import mace_polar

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alanine-dipeptide")
pdb = openmm.app.PDBFile(os.path.join(data_dir, "alanine-dipeptide-explicit.pdb"))
ml_atoms = [atom.index for atom in pdb.topology.atoms() if atom.residue.chain.index == 0]
mm_atoms = [atom.index for atom in pdb.topology.atoms() if atom.residue.chain.index != 0]
positions = pdb.positions.value_in_unit(unit.angstrom)

calc = mace_polar("polar-1-s", device="cpu", default_dtype="float64")
model = calc.models[0]
# PolarMACE has two spin channels; each receives half of the external potential.
model.electric_potential_descriptor = GTOElectrostaticExternalSourceFeatures.from_features(
    model.electric_potential_descriptor, external_scale=0.5
)
model.coulomb_energy = GTOElectrostaticExternalSourceEnergy.from_energy(model.coulomb_energy)

for periodic in (False, True):
    mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    mm_system = mm_force_field.createSystem(
        pdb.topology, nonbondedMethod=openmm.app.PME if periodic else openmm.app.NoCutoff
    )
    nonbonded = [f for f in mm_system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    mm_charges = [nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in mm_atoms]

    # MM part: mechanical embedding with the ML charges zeroed, ML force in group 1 and excluded.

    ml_system = openmmml.MLPotential("mace-off23-small").createMixedSystem(
        pdb.topology, mm_system, ml_atoms, forceGroup=1
    )
    for force in ml_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for i in ml_atoms:
                _, sigma, epsilon = force.getParticleParameters(i)
                force.setParticleParameters(i, 0, sigma, epsilon)
    context = openmm.Context(ml_system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatform("Reference"))
    context.setPositions(pdb.positions)
    mm_energy = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    # ML part: PolarMACE on the ML atoms in the field of the MM point charges.

    atoms = ase.Atoms(
        symbols=[atom.element.symbol for atom in pdb.topology.atoms() if atom.index in set(ml_atoms)],
        positions=np.array([positions[i] for i in ml_atoms]),
        cell=np.array(pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.angstrom)) if periodic else None,
        pbc=periodic,
    )
    atoms.info.update(charge=0, spin=1)
    data = calc._atoms_to_batch(atoms).to_dict()
    dtype = data["positions"].dtype

    mm_feats = torch.zeros((len(mm_atoms), (int(model.atomic_multipoles_max_l) + 1) ** 2), dtype=dtype)
    mm_feats[:, 0] = torch.tensor(mm_charges, dtype=dtype)
    if getattr(model, "_charges_to_mul_ir", None) is not None:
        mm_feats = model._charges_to_mul_ir(mm_feats)
    external = dict(
        external_feats=mm_feats,
        external_positions=torch.tensor(np.array([positions[i] for i in mm_atoms]), dtype=dtype),
        external_batch=torch.zeros(len(mm_atoms), dtype=torch.long),
    )
    model.electric_potential_descriptor.set_external_sources(**external)
    model.coulomb_energy.set_external_sources(**external)
    try:
        ml_energy = float(model(data, compute_force=False)["energy"].detach())
    finally:
        model.electric_potential_descriptor.clear_external_sources()
        model.coulomb_energy.clear_external_sources()
    # eV -> kJ/mol with the factor OpenMM-ML's MACE interface uses, so that only the embedding is compared
    ml_energy = ml_energy * 96.4853

    print(f"{'periodic' if periodic else 'non-periodic'}: {mm_energy + ml_energy}")
