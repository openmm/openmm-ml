# This script computes reference energies and forces for the AceFF foundation models.

import ase.io
from huggingface_hub import hf_hub_download
from torchmdnet.calculators import TMDNETCalculator
from openmm.unit import kilojoules_per_mole, ev, item, angstrom, nanometer

atoms = ase.io.read('toluene/toluene.pdb')
atoms.info = {'charge':0.0}
results = {}
atoms.calc = TMDNETCalculator(model_file=hf_hub_download(repo_id='Acellera/AceFF-1.0', filename='aceff_v1.0.ckpt'))
results['aceff-1.0'] = (atoms.get_potential_energy(), *atoms.get_forces()[0])
atoms.calc = TMDNETCalculator(model_file=hf_hub_download(repo_id='Acellera/AceFF-1.1', filename='aceff_v1.1.ckpt'))
results['aceff-1.1'] = (atoms.get_potential_energy(), *atoms.get_forces()[0])
atoms.calc = TMDNETCalculator(model_file=hf_hub_download(repo_id='Acellera/AceFF-2.0', filename='aceff_v2.0.ckpt'), coulomb_cutoff=12)
results['aceff-2.0'] = (atoms.get_potential_energy(), *atoms.get_forces()[0])
atoms.calc = TMDNETCalculator(model_file=hf_hub_download(repo_id='Acellera/AceFF-2-OMOL', filename='aceff_v2_omol.ckpt'), coulomb_cutoff=12)
results['aceff-2-omol'] = (atoms.get_potential_energy(), *atoms.get_forces()[0])
atoms = ase.io.read('alanine-dipeptide/alanine-dipeptide-explicit.pdb')
atoms.info = {'charge':0.0}
atoms.calc = TMDNETCalculator(model_file=hf_hub_download(repo_id='Acellera/AceFF-2.0', filename='aceff_v2.0.ckpt'), coulomb_cutoff=12)
results['alanine-dipeptide'] = (atoms.get_potential_energy(), *atoms.get_forces()[0])
for key, (energy, fx, fy, fz) in results.items():
    energy = (energy*ev/item).value_in_unit(kilojoules_per_mole)
    fx = (fx*ev/item/angstrom).value_in_unit(kilojoules_per_mole/nanometer)
    fy = (fy*ev/item/angstrom).value_in_unit(kilojoules_per_mole/nanometer)
    fz = (fz*ev/item/angstrom).value_in_unit(kilojoules_per_mole/nanometer)
    print(f'{key!r}: ({energy}, {fx}, {fy}, {fz})')
