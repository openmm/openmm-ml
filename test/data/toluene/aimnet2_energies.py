# This script computes reference energies for AIMNet2.

import ase.io
from aimnet.calculators import AIMNet2ASE
from openmm.unit import kilojoules_per_mole, ev, item

results = {}
atoms = ase.io.read('toluene.pdb')
atoms.calc = AIMNet2ASE('aimnet2')
results['toluene'] = atoms.get_potential_energy()
atoms = ase.io.read('../alanine-dipeptide/alanine-dipeptide-explicit.pdb')
atoms.calc = AIMNet2ASE('aimnet2')
results['alanine-dipeptide-explicit'] = atoms.get_potential_energy()

# Toluene (neutral, H/C only, compatible with every family including rxn) energies
# for the other pretrained families.
for name in ['aimnet2-2025', 'aimnet2-nse', 'aimnet2-pd', 'aimnet2-rxn']:
    atoms = ase.io.read('toluene.pdb')
    atoms.calc = AIMNet2ASE(name)
    results[f'toluene ({name})'] = atoms.get_potential_energy()

for key in results:
    print(f'{key}: {(results[key]*ev/item).value_in_unit(kilojoules_per_mole)}')
