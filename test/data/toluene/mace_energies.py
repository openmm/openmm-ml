# This script computes reference energies for the MACE foundation models.

import ase.io
from mace.calculators.foundations_models import mace_off, mace_mp, mace_omol
from openmm.unit import kilojoules_per_mole, ev, item

atoms = ase.io.read('toluene.pdb')
results = {}
atoms.calc = mace_off('small')
results['mace-off23-small'] = atoms.get_potential_energy()
atoms.calc = mace_off('medium')
results['mace-off23-medium'] = atoms.get_potential_energy()
atoms.calc = mace_off('large')
results['mace-off23-large'] = atoms.get_potential_energy()
atoms.calc = mace_off('https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true')
results['mace-off24-medium'] = atoms.get_potential_energy()
atoms.calc = mace_mp('medium-mpa-0')
results['mace-mpa-0-medium'] = atoms.get_potential_energy()
atoms.calc = mace_mp('small-omat-0')
results['mace-omat-0-small'] = atoms.get_potential_energy()
atoms.calc = mace_mp('medium-omat-0')
results['mace-omat-0-medium'] = atoms.get_potential_energy()
atoms.calc = mace_omol('extra_large')
results['mace-omol-0-extra-large'] = atoms.get_potential_energy()
for key in results:
    print(f'{key}: {(results[key]*ev/item).value_in_unit(kilojoules_per_mole)}')
