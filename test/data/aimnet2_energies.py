# This script computes reference energies and forces for AIMNet2.

import ase.io
from aimnet.calculators import AIMNet2ASE
from openmm.unit import kilojoules_per_mole, ev, item, angstrom, nanometer

results = {}

for name in ['aimnet2', 'aimnet2-2025', 'aimnet2-nse', 'aimnet2-pd', 'aimnet2-rxn']:
    atoms = ase.io.read('toluene/toluene.pdb')
    atoms.calc = AIMNet2ASE(name)
    results[f'toluene ({name})'] = (atoms.get_potential_energy(), *atoms.get_forces()[0])

atoms = ase.io.read('alanine-dipeptide/alanine-dipeptide-explicit.pdb')
atoms.calc = AIMNet2ASE('aimnet2')
results['alanine-dipeptide-explicit'] = (atoms.get_potential_energy(), *atoms.get_forces()[0])

for key, (energy, fx, fy, fz) in results.items():
    energy = (energy*ev/item).value_in_unit(kilojoules_per_mole)
    fx = (fx*ev/item/angstrom).value_in_unit(kilojoules_per_mole/nanometer)
    fy = (fy*ev/item/angstrom).value_in_unit(kilojoules_per_mole/nanometer)
    fz = (fz*ev/item/angstrom).value_in_unit(kilojoules_per_mole/nanometer)
    print(f'{key}: ({energy}, {fx}, {fy}, {fz})')
