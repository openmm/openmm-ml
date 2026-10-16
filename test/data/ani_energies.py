# This script computes reference energies for ANI models.

import ase.io
import torchani.models
from openmm.unit import kilojoules_per_mole, ev, item, angstrom, nanometer

results = {}
atoms = ase.io.read('toluene/toluene.pdb')
atoms.calc = torchani.models.ANI1ccx().ase()
results['ani1ccx'] = (atoms.get_potential_energy(), *atoms.get_forces()[0])
atoms.calc = torchani.models.ANI2x().ase()
results['ani2x'] = (atoms.get_potential_energy(), *atoms.get_forces()[0])

atoms = ase.io.read('alanine-dipeptide/alanine-dipeptide-explicit.pdb')
atoms.calc = torchani.models.ANI2x().ase()
results['alanine-dipeptide'] = (atoms.get_potential_energy(), *atoms.get_forces()[0])

for key, (energy, fx, fy, fz) in results.items():
    energy = (energy*ev/item).value_in_unit(kilojoules_per_mole)
    fx = (fx*ev/item/angstrom).value_in_unit(kilojoules_per_mole/nanometer)
    fy = (fy*ev/item/angstrom).value_in_unit(kilojoules_per_mole/nanometer)
    fz = (fz*ev/item/angstrom).value_in_unit(kilojoules_per_mole/nanometer)
    print(f'{key}: ({energy}, {fx}, {fy}, {fz})')
