# This script computes reference energies for the FeNNix models.
# You must download them from https://github.com/FeNNol-tools/FeNNol-PMC/tree/main/FENNIX-BIO1
# (the OpenMM-ML wrapper does this automatically but the FeNNol library used here does not).

import ase.io
from fennol.ase import FENNIXCalculator
from openmm.unit import kilojoules_per_mole, ev, item

# FeNNix tests comparing to reference values use 64-bit floating-point since otherwise
# there seems to be a lot of (non-deterministic) numerical noise on some platforms.

atoms = ase.io.read('toluene/toluene.pdb')
results = {}
atoms.calc = FENNIXCalculator('fennix-bio1S.fnx', use_float64=True)
results['fennix-bio1-small'] = atoms.get_potential_energy()
atoms.calc = FENNIXCalculator('fennix-bio1M.fnx', use_float64=True)
results['fennix-bio1-medium'] = atoms.get_potential_energy()

atoms = ase.io.read('methanol-ions/methanol-ions.pdb')
atoms.set_initial_charges([0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1])
atoms.calc = FENNIXCalculator('fennix-bio1S.fnx', use_float64=True)
results['fennix-bio1-small (charged)'] = atoms.get_potential_energy()
atoms.calc = FENNIXCalculator('fennix-bio1M.fnx', use_float64=True)
results['fennix-bio1-medium (charged)'] = atoms.get_potential_energy()
atoms.calc = FENNIXCalculator('fennix-bio1S-finetuneIons.fnx', use_float64=True)
results['fennix-bio1-small-finetune-ions (charged)'] = atoms.get_potential_energy()
atoms.calc = FENNIXCalculator('fennix-bio1M-finetuneIons.fnx', use_float64=True)
results['fennix-bio1-medium-finetune-ions (charged)'] = atoms.get_potential_energy()

atoms = ase.io.read('alanine-dipeptide/alanine-dipeptide-explicit.pdb')
atoms.calc = FENNIXCalculator('fennix-bio1S.fnx', use_float64=True)
results['alanine-dipeptide'] = atoms.get_potential_energy()
for key in results:
    print(f'{key}: {(results[key]*ev/item).value_in_unit(kilojoules_per_mole)}')
