# This script computes reference energies for the AceFF foundation models.

import ase.io
from huggingface_hub import hf_hub_download
from torchmdnet.calculators import TMDNETCalculator
from openmm.unit import kilojoules_per_mole, ev, item

atoms = ase.io.read('toluene/toluene.pdb')
atoms.info = {'charge':0.0}
results = {}
atoms.calc = TMDNETCalculator(model_file=hf_hub_download(repo_id='Acellera/AceFF-1.0', filename='aceff_v1.0.ckpt'))
results['aceff-1.0'] = atoms.get_potential_energy()
atoms.calc = TMDNETCalculator(model_file=hf_hub_download(repo_id='Acellera/AceFF-1.1', filename='aceff_v1.1.ckpt'))
results['aceff-1.1'] = atoms.get_potential_energy()
atoms.calc = TMDNETCalculator(model_file=hf_hub_download(repo_id='Acellera/AceFF-2.0', filename='aceff_v2.0.ckpt'), coulomb_cutoff=12)
results['aceff-2.0'] = atoms.get_potential_energy()
atoms = ase.io.read('alanine-dipeptide/alanine-dipeptide-explicit.pdb')
atoms.info = {'charge':0.0}
atoms.calc = TMDNETCalculator(model_file=hf_hub_download(repo_id='Acellera/AceFF-2.0', filename='aceff_v2.0.ckpt'), coulomb_cutoff=12)
results['alanine-dipeptide'] = atoms.get_potential_energy()
for key in results:
    print(f'{key}: {(results[key]*ev/item).value_in_unit(kilojoules_per_mole)}')
