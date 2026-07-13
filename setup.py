"""OpenMM-ML: A high level API for using machine learning models in OpenMM simulations
"""

from setuptools import setup, find_packages

DOCLINES = __doc__.split("\n")

########################
__version__ = '1.7'
VERSION = __version__
ISRELEASED = False
########################
CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: Bio-Informatics
Topic :: Scientific/Engineering :: Chemistry
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""


setup(
    name='openmmml',
    author='Peter Eastman',
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    version=__version__,
    license='MIT',
    url='https://github.com/openmm/openmm-ml',
    platforms=['Linux', 'Mac OS-X', 'Unix', 'Windows'],
    classifiers=CLASSIFIERS.splitlines(),
    packages=find_packages(),
    zip_safe=False,
    install_requires=['numpy', 'openmm >= 8.5.2'],
    entry_points={
        'openmmml.potentials': [
            # Custom AIMNet2 model supplied by path.
            'aimnet = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            # AIMNet2 registry models: the canonical name and family alias for each
            # supported model from the aimnet model registry
            # (aimnet/calculators/model_registry.yaml) is passed straight through to
            # AIMNet2Calculator, which resolves the family and ensemble member.  The
            # trailing _0.._3 selects the ensemble member; the short alias for each
            # family (e.g. 'aimnet2') maps to member 0.  Legacy underscore aliases from
            # the registry (e.g. 'aimnet2_rxn_0') are intentionally not registered.
            # wb97m-d3 family.
            'aimnet2-wb97m-d3_0 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-wb97m-d3_1 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-wb97m-d3_2 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-wb97m-d3_3 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-wb97m = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            # b973c-2025-d3 family.
            'aimnet2-b973c-2025-d3_0 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-b973c-2025-d3_1 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-b973c-2025-d3_2 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-b973c-2025-d3_3 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-2025 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            # nse family (open-shell systems and radicals).
            'aimnet2-nse_0 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-nse_1 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-nse_2 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-nse_3 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-nse = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            # pd family (Pd catalysis).
            'aimnet2-pd_0 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-pd_1 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-pd_2 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-pd_3 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-pd = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            # rxn family (reactive chemistry; net-neutral, H/C/N/O only).
            'aimnet2-rxn_0 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-rxn_1 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-rxn_2 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-rxn_3 = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'aimnet2-rxn = openmmml.models.aimnet2potential:AIMNet2PotentialImplFactory',
            'ani1ccx = openmmml.models.anipotential:ANIPotentialImplFactory',
            'ani2x = openmmml.models.anipotential:ANIPotentialImplFactory',
            'ase = openmmml.models.asepotential:ASEPotentialImplFactory',
            'fennix = openmmml.models.fennixpotential:FeNNixPotentialImplFactory',
            'fennix-bio1-small = openmmml.models.fennixpotential:FeNNixPotentialImplFactory',
            'fennix-bio1-medium = openmmml.models.fennixpotential:FeNNixPotentialImplFactory',
            'fennix-bio1-small-finetune-ions = openmmml.models.fennixpotential:FeNNixPotentialImplFactory',
            'fennix-bio1-medium-finetune-ions = openmmml.models.fennixpotential:FeNNixPotentialImplFactory',
            'mace = openmmml.models.macepotential:MACEPotentialImplFactory',
            'mace-off23-small = openmmml.models.macepotential:MACEPotentialImplFactory',
            'mace-off23-medium = openmmml.models.macepotential:MACEPotentialImplFactory',
            'mace-off23-large = openmmml.models.macepotential:MACEPotentialImplFactory',
            'mace-off24-medium = openmmml.models.macepotential:MACEPotentialImplFactory',
            'mace-mpa-0-medium = openmmml.models.macepotential:MACEPotentialImplFactory',
            'mace-omat-0-small = openmmml.models.macepotential:MACEPotentialImplFactory',
            'mace-omat-0-medium = openmmml.models.macepotential:MACEPotentialImplFactory',
            'mace-omol-0-extra-large = openmmml.models.macepotential:MACEPotentialImplFactory',
            'nequip = openmmml.models.nequippotential:NequIPPotentialImplFactory',
            'orb-v3-conservative-inf-omat = openmmml.models.orbpotential:OrbPotentialImplFactory',
            'orb-v3-conservative-omol = openmmml.models.orbpotential:OrbPotentialImplFactory',
            'deepmd = openmmml.models.deepmdpotential:DeepmdPotentialImplFactory',
            'torchmdnet = openmmml.models.torchmdnetpotential:TorchMDNetPotentialImplFactory',
            'aceff-1.0 = openmmml.models.torchmdnetpotential:TorchMDNetPotentialImplFactory',
            'aceff-1.1 = openmmml.models.torchmdnetpotential:TorchMDNetPotentialImplFactory',
            'aceff-2.0 = openmmml.models.torchmdnetpotential:TorchMDNetPotentialImplFactory',
        ],
        'openmmml.embeddings': [
            'mechanical = openmmml.embeddings.mechanicalembedding:MechanicalEmbeddingFactory',
        ]
    }
)
