"""
torchmdnetpotential.py: Implements the TorchMD-Net potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021-2026 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Stephen Farr

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import openmm
from openmm import unit
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory
from typing import Iterable, Optional
import numpy as np
try:
    import torch
except ImportError:
    # PyTorch is only needed when a TorchMD-Net potential is used.
    torch = None

# 1/(4 pi eps0) in kJ/mol nm / e^2, as used by OpenMM.
ONE_4PI_EPS0 = 138.935456

# Default atomic polarizabilities for the electrostatic embedding's induction term,
# keyed by atomic number, in bohr^3.  These are the values used with ANI-MBIS in
# J. A. Semelak et al., arXiv:2409.17852 (https://arxiv.org/abs/2409.17852).
BOHR3_TO_NM3 = 0.052917721**3
DEFAULT_POLARIZABILITIES = {1: 3.08, 6: 11.30, 7: 7.40, 8: 5.30, 9: 3.74, 16: 19.40, 17: 14.60}

# Default dielectric constant screening the induction energy.
DEFAULT_EPSILON = 2.0

class TorchMDNetPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates TorchMDNetPotentialImpl objects."""

    def createImpl(
        self, 
        name: str, 
        modelPath: Optional[str] = None,
        lengthScale: float = 0.1, # angstrom -> nm
        energyScale: float = 96.4916,  # eV -> kJ/mol
    ) -> MLPotentialImpl:
        return TorchMDNetPotentialImpl(name, modelPath, lengthScale, energyScale)

class TorchMDNetPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the TorchMDNet potential.

    The TorchMDNet potential is constructed using `torchmdnet` to build a PyTorch model,
    and then integrated into the OpenMM System using a TorchForce.  To use it, specify the model by name
    and provide the path to a model.

    >>> potential = MLPotential('torchmdnet', modelPath=<model_file_path>)

    The default energy and length scales assume a model is trained with positions in angstroms and energies in eV.
    If this is not the case you can specify the length and energy scales by passing the factors that convert the model
    distance to nm and the energy to kJ/mol, for example:

    >>> potential = MLPotential('torchmdnet', modelPath=<model_file_path>, 
                                lengthScale=0.1 # angstrom to nm, 
                                energyScale=4.184 # kcal/mol to kJ/mol)

    During system creation you can enable CUDA graphs for a speed-up for small molecules:

    >>>  system = potential.createSystem(pdb.topology, cudaGraphs=True)

    The default is to enable this for TensorNet models.

    You can also specify the molecule's total charge:

    >>>  system = potential.createSystem(pdb.topology, charge=0)

    Pretained AceFF models can be used directly:

    >>> potential = MLPotential('aceff-2-resp-1')

    >>> potential = MLPotential('aceff-2-omol')

    >>> potential = MLPotential('aceff-2.0')

    >>> potential = MLPotential('aceff-1.1')

    >>> potential = MLPotential('aceff-1.0')

    Electrostatic embedding
    -----------------------
    'aceff-2-resp-1' also predicts per-atom RESP charges and supports an electrostatic
    embedding for mixed ML/MM systems:

    >>> system = potential.createMixedSystem(topology, mmSystem, mlAtoms, embedding='electrostatic')

    The MM System must contain one NonbondedForce using PME, and the ML region must not
    be bonded to the MM region.  The System is built as for the mechanical embedding
    (the force field computes ML-MM interactions with its fixed charges q_FF), and the ML
    force adds the following terms over ML atoms i and MM atoms j within the PME cutoff:

    - a Coulomb correction sum_ij (q_ML,i - q_FF,i) q_j erfc(alpha r_ij) / r_ij, where
      q_ML are the charges predicted by the model and alpha is the PME screening
      parameter (direct space only);
    - an induction energy -1/(2 epsilon) sum_i a_i |F_i|^2, where F_i is the
      erfc-screened electric field of the MM charges at ML atom i and a_i its
      polarizability.

    These optional arguments to createMixedSystem() control the induction term:

    - ``epsilon`` (default 2.0): dielectric constant screening the induction energy.
    - ``polarizabilities``: dict mapping elements (Element, symbol, or atomic number) to
      polarizabilities (Quantities with volume units, or floats in nm^3).  These override
      the defaults (ANI-MBIS values for H, C, N, O, F, S, Cl).
    - ``tholeDamping`` (default None): a Thole linear damping parameter (e.g. 1.3).  If
      given, the field is Thole damped; by default it is not damped.  Farr and De
      Fabritiis (arXiv:2608.13355) use the damping with 1.3 because it keeps the
      induction term stable in alchemical transformations such as the Alchemical
      Transfer Method, where MM charges can come very close to the ML atoms.  It is not
      needed for normal simulations.

    References
    ----------
    - The electrostatic embedding and its use for relative binding free energies with
      AceFF-2-RESP-1: S. E. Farr and G. De Fabritiis, "Evaluating Electrostatic Embedding
      MLIP/MM for Relative Binding Free Energy Calculations", arXiv:2608.13355 (2026),
      https://arxiv.org/abs/2608.13355
    - The embedding scheme (Coulomb term with ML charges plus a polarization correction)
      and the default polarizabilities, from ANI-MBIS: J. A. Semelak et al., "ANI Neural
      Networks Meet Electrostatics: A ML/MM Implementation in Amber", arXiv:2409.17852
      (2024), https://arxiv.org/abs/2409.17852

    Coulomb cutoff behavior
    ------------------------
    The Coulomb cutoff in TorchMD-Net uses a reaction-field approximation. Applying it to a
    non-periodic system introduces errors, so by default the cutoff is only used when the
    system uses periodic boundary conditions.

    You can override this with the ``useCoulombCutoff`` argument if you know which behavior
    you want, for example:

    >>>  system = potential.createSystem(pdb.topology, useCoulombCutoff=False)

    """

    # (Repository ID, filename, long-range)
    KNOWN_MODELS = {
        'aceff-1.0': ('Acellera/AceFF-1.0', 'aceff_v1.0.ckpt', False),
        'aceff-1.1': ('Acellera/AceFF-1.1', 'aceff_v1.1.ckpt', False),
        'aceff-2.0': ('Acellera/AceFF-2.0', 'aceff_v2.0.ckpt', False),
        'aceff-2-omol': ('Acellera/AceFF-2-OMOL', 'aceff_v2_omol.ckpt', False),
        'aceff-2-resp-1': ('Acellera/AceFF-2-RESP-1', 'aceff-2-resp-1.ckpt', False),
    }

    # Models that predict charges, and the architecture to load their checkpoints with.
    CHARGE_MODELS = {
        'aceff-2-resp-1': 'tensornet2_resp',
    }

    def __init__(self, 
                 name: str,
                 modelPath: str,
                 lengthScale: float,
                 energyScale: float
    ) -> None:
        """
        Initialize the TorchMDNetPotentialImpl.

        Parameters
        ----------
        name : str
            The name of the model.
            'torchmdnet' for a local model file, or pretrained models are available: 'aceff-1.0', 'aceff-1.1', 'aceff-2.0' or 'aceff-2-omol'. 
        modelPath : str, optional
            The path to the locally trained torchmdnet model if ``name`` is 'torchmdnet'.
        lengthScale : float
            The length conversion factor from the model units to nanometers. 
            If not specified the default is 0.1 which corresponds to a model in angstrom
        energyScale : float
            The energy conversion factor from the model units to kJ/mol.
            If not specified the default is 96.4916 which corresponds to a model in eV.
   
        """
        self.name = name
        self.modelPath = modelPath
        self.lengthScale = lengthScale
        self.energyScale = energyScale

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  **args):
        # Load the TorchMDNet model.
        try:
            import torchmdnet
            from torchmdnet.models.model import load_model
        except ImportError as e:
            raise ImportError(f"Failed to import torchmdnet please install from https://torchmd-net.readthedocs.io/en/latest/installation.html")
        import torch

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        device = self._getTorchDevice(args)
        numbers = torch.tensor([atom.element.atomic_number for atom in includedAtoms], device=device, requires_grad=False)
        charge = torch.tensor([args.get('charge', 0)], dtype=torch.float32, device=device, requires_grad=False)
        cutoff = 10*args.get('coulomb_cutoff', 1.2)
        if unit.is_quantity(cutoff):
            cutoff = cutoff.value_in_unit(unit.angstrom)

        if self.name == 'torchmdnet':
            # a local path to a torchmdnet checkpoint must be provided 
            model_file_path = self.modelPath
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise ImportError(f"Failed to import huggingface_hub please install from https://huggingface.co/docs/huggingface_hub/en/installation")

            if self.name in TorchMDNetPotentialImpl.KNOWN_MODELS:
                repo_id, filename, _ = TorchMDNetPotentialImpl.KNOWN_MODELS[self.name]
            else:
                raise ValueError(f'Model name {self.name} does not exist.')

            model_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )

        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()
        use_coulomb_cutoff = args.get('useCoulombCutoff', periodic)
        load_args = {}
        if self.name in TorchMDNetPotentialImpl.CHARGE_MODELS:
            load_args['model'] = TorchMDNetPotentialImpl.CHARGE_MODELS[self.name]
        model = load_model(
            model_file_path,
            **load_args,
            derivative=False,
            remove_ref_energy = args.get('remove_ref_energy', True),
            max_num_neighbors = min(args.get('max_num_neighbors', 64), numbers.shape[0]),
            coulomb_cutoff = cutoff if use_coulomb_cutoff else None,
            static_shapes = True,
        ).to(device)
        for parameter in model.parameters():
            parameter.requires_grad = False
        batch = args.get('batch', None)
        if batch is None:
            batch = torch.zeros_like(numbers, requires_grad=False)
        else:
            batch = torch.tensor(batch, dtype=torch.long, device=device, requires_grad=False)

        # Create the PythonForce and add it to the System.

        electrostatic = args.get('_electrostatic', None)
        if electrostatic is None:
            compute = _ComputeTorchMDNet(model=model,
                                         numbers=numbers,
                                         charge=charge,
                                         batch=batch,
                                         lengthScale=self.lengthScale,
                                         energyScale=self.energyScale,
                                         periodic=periodic)
        else:
            # The electrostatic embedding needs the positions of all particles.
            compute = _ComputeTorchMDNetElectrostatic(model=model,
                                                      numbers=numbers,
                                                      charge=charge,
                                                      batch=batch,
                                                      lengthScale=self.lengthScale,
                                                      energyScale=self.energyScale,
                                                      mlIndices=np.array(atoms),
                                                      **electrostatic)
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        if atoms is not None and electrostatic is None:
            force.setParticles(atoms)
        system.addForce(force)

    def getMLLongRange(self) -> bool | None:
        if self.name in TorchMDNetPotentialImpl.KNOWN_MODELS:
            _, _, longRange = TorchMDNetPotentialImpl.KNOWN_MODELS[self.name]
            return longRange
        return None

    def getSupportedEmbeddings(self) -> list[str]:
        if self.name in TorchMDNetPotentialImpl.CHARGE_MODELS:
            return ['electrostatic']
        return []

    def createMixedSystem(self,
                          topology: openmm.app.Topology,
                          system: openmm.System,
                          atoms: list[int],
                          forceGroup: int,
                          interpolate: bool,
                          embedding: str,
                          **args) -> openmm.System:
        if embedding != 'electrostatic':
            raise ValueError(f'Unsupported embedding: {embedding}')
        from openmmml.embeddings.mechanicalembedding import MechanicalEmbedding

        # Read the PME settings and the charges of all particles from the MM System.

        nonbonded = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
        if len(nonbonded) != 1:
            raise ValueError('The electrostatic embedding requires the System to contain exactly one NonbondedForce.')
        nonbonded = nonbonded[0]
        if nonbonded.getNonbondedMethod() != openmm.NonbondedForce.PME:
            raise ValueError('The electrostatic embedding requires the NonbondedForce to use PME.')
        cutoff = nonbonded.getCutoffDistance().value_in_unit(unit.nanometer)
        alpha = nonbonded.getPMEParameters()[0]
        if unit.is_quantity(alpha):
            alpha = alpha.value_in_unit(unit.nanometer**-1)
        if alpha == 0:
            # Not set explicitly: use the value OpenMM chooses.
            alpha = np.sqrt(-np.log(2*nonbonded.getEwaldErrorTolerance()))/cutoff
        charges = np.array([nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                            for i in range(system.getNumParticles())])

        atomSet = set(atoms)
        for i in range(nonbonded.getNumExceptions()):
            p1, p2, _, _, _ = nonbonded.getExceptionParameters(i)
            if (p1 in atomSet) != (p2 in atomSet):
                raise ValueError('The electrostatic embedding requires the ML region not to be bonded to the MM region.')
        mmIndices = np.array([i for i in range(system.getNumParticles()) if i not in atomSet])

        # Polarizabilities (nm^3) of the ML atoms: defaults, overridden by the user.

        table = {z: a*BOHR3_TO_NM3 for z, a in DEFAULT_POLARIZABILITIES.items()}
        for key, value in args.get('polarizabilities', {}).items():
            if isinstance(key, openmm.app.Element):
                key = key.atomic_number
            elif isinstance(key, str):
                key = openmm.app.Element.getBySymbol(key).atomic_number
            if unit.is_quantity(value):
                value = value.value_in_unit(unit.nanometer**3)
            table[int(key)] = float(value)
        allAtoms = list(topology.atoms())
        polarizabilities = []
        for i in atoms:
            element = allAtoms[i].element
            if element.atomic_number not in table:
                raise ValueError(f'No polarizability is defined for element {element.symbol}.  Specify it with the polarizabilities argument.')
            polarizabilities.append(table[element.atomic_number])

        electrostatic = dict(mmIndices=mmIndices,
                             mlFFCharges=charges[atoms],
                             mmCharges=charges[mmIndices],
                             polarizabilities=np.array(polarizabilities),
                             alpha=float(alpha),
                             cutoff=float(cutoff),
                             epsilon=float(args.get('epsilon', DEFAULT_EPSILON)),
                             tholeDamping=args.get('tholeDamping', None))

        # The rest is the mechanical embedding; addForces() receives the parameters above.

        return MechanicalEmbedding().createMixedSystem(self, topology, system, atoms, forceGroup, interpolate,
                                                       _electrostatic=electrostatic, **args)

class _ComputeTorchMDNet(object):
    def __init__(self, model, numbers, charge, batch, lengthScale, energyScale, periodic):
        self.model = model
        self.compiled_model = None
        self.numbers = numbers
        self.charge = charge
        self.batch = batch
        self.lengthScale = lengthScale
        self.energyScale = energyScale
        self.periodic = periodic
        self.has_recompiled = False

    def __call__(self, state):
        import torch
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = torch.tensor(positions, dtype=torch.float32, device=self.numbers.device)
        positions.requires_grad_(True)
        if self.periodic:
            cell = torch.tensor(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer), dtype=torch.float32, device=self.numbers.device)/self.lengthScale
        else:
            cell = None
        if self.compiled_model is None:
            # The model can't be compiled until after it has been invoked once.
            energy = self.model(z=self.numbers, pos=positions/self.lengthScale, batch=self.batch, q=self.charge, box=cell)[0]*self.energyScale
            self.compiled_model = torch.compile(self.model, backend="inductor", dynamic=False, fullgraph=True, mode="reduce-overhead")
        else:
            try:
                energy = self.compiled_model(z=self.numbers, pos=positions/self.lengthScale, batch=self.batch, q=self.charge, box=cell)[0]*self.energyScale
            except AssertionError:
                if self.has_recompiled:
                    raise

                # Workaround for https://github.com/openmm/openmm-ml/issues/128.  Threading bugs in torch.compile can cause it to
                # incorrectly reuse cached components from earlier models.  The solution is to compile with mode="default".

                self.has_recompiled = True
                self.compiled_model = torch.compile(self.model, backend="inductor", dynamic=False, fullgraph=True, mode="default")
                energy = self.compiled_model(z=self.numbers, pos=positions/self.lengthScale, batch=self.batch, q=self.charge, box=cell)[0]*self.energyScale
        energy.backward()
        forces = (-positions.grad).detach().cpu().numpy()
        return energy, forces

class TorchMDNetElectrostaticEnergy(torch.nn.Module if torch is not None else object):
    """Wraps a TorchMD-Net model that predicts charges (e.g. AceFF-2-RESP-1).  forward()
    maps (positions of all particles, box vectors), both in nm, to the ML energy plus the
    electrostatic embedding terms in kJ/mol (see the TorchMDNetPotentialImpl docstring;
    Farr and De Fabritiis, arXiv:2608.13355, following Semelak et al., arXiv:2409.17852).
    Calling backward() on the energy gives the forces on all particles."""

    def __init__(self, model, numbers, charge, batch, lengthScale, energyScale, mlIndices, mmIndices,
                 mlFFCharges, mmCharges, polarizabilities, alpha, cutoff, epsilon, tholeDamping):
        super().__init__()
        device = numbers.device
        self.model = model
        self.register_buffer('numbers', numbers)
        self.register_buffer('charge', charge)
        self.register_buffer('batch', batch)
        self.register_buffer('mlIndices', torch.tensor(mlIndices, dtype=torch.long, device=device))
        self.register_buffer('mmIndices', torch.tensor(mmIndices, dtype=torch.long, device=device))
        self.register_buffer('mlFFCharges', torch.tensor(mlFFCharges, dtype=torch.float32, device=device))
        self.register_buffer('mmCharges', torch.tensor(mmCharges, dtype=torch.float32, device=device))
        self.register_buffer('polarizabilities', torch.tensor(polarizabilities, dtype=torch.float32, device=device))
        self.lengthScale = float(lengthScale)
        self.energyScale = float(energyScale)
        self.alpha = float(alpha)
        self.cutoff = float(cutoff)
        self.epsilon = float(epsilon)
        self.tholeDamping = None if tholeDamping is None else float(tholeDamping)

    def forward(self, positions, box):
        ml = positions[self.mlIndices]
        mm = positions[self.mmIndices]

        # ML energy and charges.

        out = self.model(z=self.numbers, pos=ml/self.lengthScale, batch=self.batch, q=self.charge, box=box/self.lengthScale)
        energy = out[0].sum()*self.energyScale
        qML = out[2].reshape(-1)

        # Minimum image ML-MM displacements (OpenMM's reduced triclinic box).

        delta = ml.unsqueeze(1) - mm.unsqueeze(0)
        delta = delta - torch.round(delta[..., 2:3]/box[2, 2])*box[2]
        delta = delta - torch.round(delta[..., 1:2]/box[1, 1])*box[1]
        delta = delta - torch.round(delta[..., 0:1]/box[0, 0])*box[0]
        r = torch.sqrt((delta**2).sum(-1))
        mask = r < self.cutoff
        r = torch.where(mask, r, torch.ones_like(r))  # avoids NaN gradients beyond the cutoff
        screen = torch.erfc(self.alpha*r)

        # Coulomb correction: direct space interaction of the charge difference with the MM charges.

        qq = (qML - self.mlFFCharges).unsqueeze(1)*self.mmCharges.unsqueeze(0)
        eCoulomb = ONE_4PI_EPS0*torch.where(mask, qq*screen/r, 0).sum()

        # Induction: ML atoms polarized by the erfc-screened field of the MM charges.

        kernel = screen/r**2 + (2*self.alpha/np.sqrt(np.pi))*torch.exp(-(self.alpha*r)**2)/r
        if self.tholeDamping is not None:
            s = self.tholeDamping*r/self.polarizabilities.unsqueeze(1)**(1/3)
            kernel = kernel*(1 - (1 + s + 0.5*s**2)*torch.exp(-s))
        kernel = torch.where(mask, self.mmCharges.unsqueeze(0)*kernel/r, 0)
        field = (kernel.unsqueeze(-1)*delta).sum(1)
        eInduction = -0.5*ONE_4PI_EPS0/self.epsilon*(self.polarizabilities*(field**2).sum(-1)).sum()

        return energy + eCoulomb + eInduction

class _ComputeTorchMDNetElectrostatic(object):
    """PythonForce callback for the electrostatic embedding.  The whole energy (model and
    embedding terms) is compiled with torch.compile, as in _ComputeTorchMDNet."""

    def __init__(self, **args):
        self.module = TorchMDNetElectrostaticEnergy(**args)
        self.device = args['numbers'].device
        self.compiled_model = None
        self.has_recompiled = False

    def __call__(self, state):
        positions = torch.tensor(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=torch.float32, device=self.device)
        positions.requires_grad_(True)
        box = torch.tensor(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer), dtype=torch.float32, device=self.device)
        if self.compiled_model is None:
            # The model can't be compiled until after it has been invoked once.
            energy = self.module(positions, box)
            self.compiled_model = torch.compile(self.module, backend="inductor", dynamic=False, fullgraph=True, mode="reduce-overhead")
        else:
            try:
                energy = self.compiled_model(positions, box)
            except AssertionError:
                if self.has_recompiled:
                    raise

                # Workaround for https://github.com/openmm/openmm-ml/issues/128.  Threading bugs in torch.compile can cause it to
                # incorrectly reuse cached components from earlier models.  The solution is to compile with mode="default".

                self.has_recompiled = True
                self.compiled_model = torch.compile(self.module, backend="inductor", dynamic=False, fullgraph=True, mode="default")
                energy = self.compiled_model(positions, box)
        energy.backward()
        forces = (-positions.grad).detach().cpu().numpy()
        return float(energy.detach()), forces
