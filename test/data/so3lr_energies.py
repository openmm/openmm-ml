# This script computes reference energies and forces for the SO3LR models with
# so3lr itself, in double precision and with the settings openmm-ml uses by
# default.  Run it from this directory.

import ase.io
import jax
from so3lr import So3lrCalculator, So3lrEngine
from openmm.unit import kilojoules_per_mole, ev, item, angstrom, nanometer

jax.config.update('jax_enable_x64', True)

def convert(energy, forces):
    energy = (energy * ev / item).value_in_unit(kilojoules_per_mole)
    force = (forces[0] * ev / item / angstrom).value_in_unit(kilojoules_per_mole / nanometer)
    return (energy, *force)

def compute(atoms, charge=0, **kwargs):
    atoms.info['charge'] = charge
    atoms.info['multiplicity'] = 1
    atoms.calc = So3lrCalculator(model='so3lr-tauri-m-2606', precision='float64', **kwargs)
    return convert(atoms.get_potential_energy(), atoms.get_forces())

results = {}
# Non-periodic systems: real space up to 100 A.
toluene = ase.io.read('toluene/toluene.pdb')
results['so3lr-tauri-m-2606'] = compute(toluene, lr_cutoff=100.0)

ions = ase.io.read('methanol-ions/methanol-ions.pdb')
results['charged'] = compute(ions, charge=1, lr_cutoff=100.0)

# Periodic systems: real space up to 12 A, and the default, PME electrostatics
# and dispersion with a 10 A cutoff (the ASE calculator has no k-space
# dispersion, so so3lr's neighbor-list engine computes this one).
dipeptide = ase.io.read('alanine-dipeptide/alanine-dipeptide-explicit.pdb')
dipeptide.wrap()
results['periodic'] = compute(dipeptide)
engine = So3lrEngine(dipeptide.numbers, model='so3lr-tauri-m-2606', precision='float64', engine='standard',
                     lr_cutoff=10.0, kspace_electrostatics='pme', kspace_smearing=2.0, kspace_spacing=1.55,
                     kspace_dispersion='pme', kspace_dispersion_smearing=2.0, kspace_dispersion_spacing=1.55)
results['periodic-pme'] = convert(*engine(dipeptide.positions, dipeptide.cell[:]))

for key, value in results.items():
    print(f'    {key!r}: ({", ".join(repr(float(v)) for v in value)}),')
