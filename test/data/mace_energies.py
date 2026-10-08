# This script computes reference energies for the MACE foundation models.

import ase.io
from mace.calculators.foundations_models import mace_off, mace_mp, mace_omol, mace_polar
from openmm.unit import kilojoules_per_mole, ev, item

results = {}

atoms = ase.io.read('toluene/toluene.pdb')
atoms.calc = mace_off('small')
results['toluene', 'mace-off23-small'] = atoms.get_potential_energy()
atoms.calc = mace_off('medium')
results['toluene', 'mace-off23-medium'] = atoms.get_potential_energy()
atoms.calc = mace_off('large')
results['toluene', 'mace-off23-large'] = atoms.get_potential_energy()
atoms.calc = mace_off('https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true')
results['toluene', 'mace-off24-medium'] = atoms.get_potential_energy()
atoms.calc = mace_mp('medium-mpa-0')
results['toluene', 'mace-mpa-0-medium'] = atoms.get_potential_energy()
atoms.calc = mace_mp('small-omat-0')
results['toluene', 'mace-omat-0-small'] = atoms.get_potential_energy()
atoms.calc = mace_mp('medium-omat-0')
results['toluene', 'mace-omat-0-medium'] = atoms.get_potential_energy()
atoms.calc = mace_omol('extra_large')
results['toluene', 'mace-omol-0-extra-large'] = atoms.get_potential_energy()
try:
    import les
    atoms.calc = mace_off('https://github.com/ChengUCB/les_fit/blob/main/MACELES-OFF/MACELES-OFF_small_converted.model?raw=true')
    results['toluene', 'mace-les-off-small'] = atoms.get_potential_energy()
except ImportError:
    print('No MACELES reference energies will be generated; you must first install LES from https://github.com/ChengUCB/les')
try:
    import graph_longrange
    atoms.calc = mace_polar('polar-1-s')
    results['toluene', 'mace-polar-1-small'] = atoms.get_potential_energy()
    atoms.calc = mace_polar('polar-1-m')
    results['toluene', 'mace-polar-1-medium'] = atoms.get_potential_energy()
    atoms.calc = mace_polar('polar-1-l')
    results['toluene', 'mace-polar-1-large'] = atoms.get_potential_energy()
except ImportError:
    print('No MACE-POLAR reference energies will be generated; you must first install graph_longrange from https://github.com/WillBaldwin0/graph_electrostatics')

atoms = ase.io.read('water/water.pdb')
atoms.calc = mace_off('small')
results['water', 'mace-off23-small'] = atoms.get_potential_energy()
atoms.calc = mace_off('medium')
results['water', 'mace-off23-medium'] = atoms.get_potential_energy()
atoms.calc = mace_off('https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true')
results['water', 'mace-off24-medium'] = atoms.get_potential_energy()
atoms.calc = mace_mp('medium-mpa-0')
results['water', 'mace-mpa-0-medium'] = atoms.get_potential_energy()
atoms.calc = mace_mp('small-omat-0')
results['water', 'mace-omat-0-small'] = atoms.get_potential_energy()
atoms.calc = mace_mp('medium-omat-0')
results['water', 'mace-omat-0-medium'] = atoms.get_potential_energy()
try:
    import les
    atoms.calc = mace_off('https://github.com/ChengUCB/les_fit/blob/main/MACELES-OFF/MACELES-OFF_small_converted.model?raw=true')
    results['water', 'mace-les-off-small'] = atoms.get_potential_energy()
except ImportError:
    print('No MACELES reference energies will be generated; you must first install LES from https://github.com/ChengUCB/les')
try:
    import graph_longrange
    atoms.calc = mace_polar('polar-1-s')
    results['water', 'mace-polar-1-small'] = atoms.get_potential_energy()
    atoms.calc = mace_polar('polar-1-m')
    results['water', 'mace-polar-1-medium'] = atoms.get_potential_energy()
except ImportError:
    print('No MACE-POLAR reference energies will be generated; you must first install graph_longrange from https://github.com/WillBaldwin0/graph_electrostatics')

atoms = ase.io.read('alanine-dipeptide/alanine-dipeptide-explicit.pdb')
atoms.calc = mace_off('small')
results['alanine-dipeptide', 'mace-off23-small'] = atoms.get_potential_energy()

for key in results:
    print(f'{key}: {(results[key]*ev/item).value_in_unit(kilojoules_per_mole)}')
