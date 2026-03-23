# This script computes reference energies for the Orb models.

import ase.io
from orb_models.forcefield import pretrained as orb
from orb_models.forcefield.inference.calculator import ORBCalculator
from openmm.unit import kilojoules_per_mole, ev, item

def make_calculator(name):
    orbff, atoms_adapter = orb.ORB_PRETRAINED_MODELS[name]()
    return ORBCalculator(orbff, atoms_adapter=atoms_adapter)

results = {}

atoms = ase.io.read('toluene/toluene.pdb')
atoms.info['charge'] = 0
atoms.info['spin'] = 1

for name in ['orb-v3-conservative-inf-omat', 'orb-v3-conservative-omol']:
    atoms.calc = make_calculator(name)
    results[name] = atoms.get_potential_energy()

atoms.info['charge'] = -1
atoms.info['spin'] = 3
atoms.calc = make_calculator('orb-v3-conservative-omol')
results['override-charge-spin'] = atoms.get_potential_energy()

atoms = ase.io.read('alanine-dipeptide/alanine-dipeptide-explicit.pdb')
atoms.info['charge'] = 0
atoms.info['spin'] = 1
atoms.calc = make_calculator('orb-v3-conservative-omol')
results['alanine-dipeptide'] = atoms.get_potential_energy()

for key in results:
    print(f'{key}: {(results[key]*ev/item).value_in_unit(kilojoules_per_mole)}')
