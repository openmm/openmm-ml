import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml import MLPotential

mace = pytest.importorskip("torchmdnet", reason="TorchMD-Net is not installed")
platform_ints = range(mm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestTorchMDNetPotential:
    @pytest.mark.parametrize("model", ['aceff-1.0', 'aceff-1.1', 'aceff-2.0', 'aceff-2-omol'])
    def testCreatePureMLSystem(self, platform_int, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        kwargs = {'returnEnergyType': 'energy'}
        if model in ('aceff-2.0', 'aceff-2-omol'):
            kwargs['coulomb_cutoff'] = 1.2
            kwargs['useCoulombCutoff'] = True
        system = potential.createSystem(pdb.topology, **kwargs)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        state = context.getState(energy=True, forces=True)
        energyML = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forceML = state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
        # Reference energies and forces are calculated with TMDNETCalculator
        refEnergyForce = {
            'aceff-1.0': (-6974.531656957799, 330.2444763183594, -1078.1910400390625, 784.6386108398438),
            'aceff-1.1': (-6976.570722280534, 335.0249328613281, -1081.064697265625, 788.1050415039062),
            'aceff-2.0': (-362.4612980692178, 387.4684753417969, -1090.3306884765625, 799.9639892578125),
            'aceff-2-omol': (-112.98626811365253, 325.1884765625, -1086.428955078125, 793.2885131835938),
        }
        assert np.isclose(energyML, refEnergyForce[model][0], rtol=1e-4)
        assert np.allclose(forceML, refEnergyForce[model][1:], rtol=1e-4)

    def testPeriodicSystem(self, platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        potential = MLPotential("aceff-2.0")
        system = potential.createSystem(pdb.topology, returnEnergyType='energy')
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        positionsOriginal = pdb.getPositions(asNumpy=True)
        energyRef = -72195.52076809801 # Calculated with TMDNETCalculator
        forceRef = (-1001.861328125, -498.4813232421875, -403.4123840332031)
        for i in range(3):
            positions = positionsOriginal + i * 0.9 * unit.nanometers # translate molecule to test PBC
            context.setPositions(positions)
            state = context.getState(energy=True, forces=True)
            energyML = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            forceML = state.getForces(asNumpy=True)[0].value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
            assert np.isclose(energyML, energyRef, rtol=1e-4)
            assert np.allclose(forceML, forceRef, rtol=1e-3)

    @pytest.mark.parametrize("model", ['aceff-1.1', 'aceff-2-resp-1'])
    def testCreateMixedSystem(self, platform_int, model):
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))
        mlAtoms = list(range(15))
        mmSystem = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential(model)
        mixedSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, interpolate=False)
        interpSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, interpolate=True)
        platform = mm.Platform.getPlatform(platform_int)
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), platform)
        mixedContext = mm.Context(mixedSystem, mm.VerletIntegrator(0.001), platform)
        interpContext = mm.Context(interpSystem, mm.VerletIntegrator(0.001), platform)
        mmContext.setPositions(inpcrd.positions)
        mixedContext.setPositions(inpcrd.positions)
        interpContext.setPositions(inpcrd.positions)
        mmEnergy = mmContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        mixedEnergy = mixedContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpEnergy1 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpContext.setParameter('lambda_interpolate', 0)
        interpEnergy2 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert np.isclose(mixedEnergy, interpEnergy1, rtol=1e-5)
        assert np.isclose(mmEnergy, interpEnergy2, rtol=1e-5)


# Electrostatic embedding of aceff-2-resp-1, tested on toluene in TIP3P water.
# Method: S. E. Farr and G. De Fabritiis, arXiv:2608.13355, following the ANI-MBIS
# embedding of J. A. Semelak et al., arXiv:2409.17852 (source of the polarizabilities).

BOHR3_TO_NM3 = 0.052917721**3
REFERENCE_POLARIZABILITIES = {'H': 3.08*BOHR3_TO_NM3, 'C': 11.30*BOHR3_TO_NM3}  # ANI-MBIS, nm^3
ONE_4PI_EPS0 = 138.935456

def _loadTolueneExplicit(nonbondedMethod=app.PME):
    prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))
    inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))
    mmSystem = prmtop.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=1.0*unit.nanometer)
    return prmtop, inpcrd, mmSystem, list(range(15))

# AceFF-2-RESP-1 charges (e) of the toluene atoms in toluene-explicit.rst7, calculated
# with TorchMD-Net directly (load_model(..., model='tensornet2_resp'), positions and box
# in angstrom, total charge 0).
REFERENCE_RESP_CHARGES = np.array([
    -0.21203146874904633, 0.10292987525463104, -0.1267547905445099, -0.17719489336013794,
    -0.12571541965007782, -0.17315566539764404, -0.14510978758335114, 0.06204691529273987,
    0.06385692954063416, 0.0645725205540657, 0.11917456239461899, 0.14130568504333496,
    0.1363997459411621, 0.13955023884773254, 0.13012579083442688])

def _referenceEmbeddingEnergy(positions, box, mlAtoms, qML, charges, alpha, cutoff, polarizabilities, epsilon, tholeDamping):
    """Coulomb correction + induction energy (kJ/mol) in float64, for a rectangular box."""
    import math
    erfc = np.vectorize(math.erfc)
    mmAtoms = np.array([i for i in range(len(positions)) if i not in set(mlAtoms)])
    boxLengths = np.diag(box)
    delta = positions[mlAtoms][:, None, :] - positions[mmAtoms][None, :, :]
    delta -= boxLengths*np.round(delta/boxLengths)
    r = np.linalg.norm(delta, axis=-1)
    mask = r < cutoff
    qMM = charges[mmAtoms]
    dq = qML - charges[mlAtoms]
    eCoulomb = ONE_4PI_EPS0*np.sum(np.where(mask, dq[:, None]*qMM[None, :]*erfc(alpha*r)/r, 0))
    kernel = erfc(alpha*r)/r**2 + 2*alpha/np.sqrt(np.pi)*np.exp(-(alpha*r)**2)/r
    if tholeDamping is not None:
        s = tholeDamping*r/polarizabilities[:, None]**(1/3)
        kernel *= 1 - (1 + s + 0.5*s**2)*np.exp(-s)
    field = np.sum(np.where(mask, qMM[None, :]*kernel/r, 0)[..., None]*delta, axis=1)
    eInduction = -0.5*ONE_4PI_EPS0/epsilon*np.sum(polarizabilities*np.sum(field**2, axis=-1))
    return eCoulomb + eInduction

def _groupEnergy(context, group):
    return context.getState(getEnergy=True, groups={group}).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestTorchMDNetElectrostaticEmbedding:

    def testTotalEnergy(self, platform_int):
        """Total potential energy of toluene in water with the electrostatic embedding
        (default options)."""
        prmtop, inpcrd, mmSystem, mlAtoms = _loadTolueneExplicit()
        potential = MLPotential('aceff-2-resp-1')
        system = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, embedding='electrostatic')
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(inpcrd.positions)
        context.setPeriodicBoxVectors(*inpcrd.boxVectors)
        energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        # Calculated with the Reference platform (PyTorch on CPU).  The ML group alone is
        # -437.56640625 kJ/mol, of which -3.4023 kJ/mol is the Coulomb correction plus
        # induction (the mechanical embedding gives -434.1640625 kJ/mol).
        refEnergy = -108519.49394475727
        assert np.isclose(refEnergy, energy, rtol=1e-5)

    @pytest.mark.parametrize("options", [
        {},
        {'epsilon': 4.0},
        {'tholeDamping': 1.3},
        {'polarizabilities': {'C': 1.5*unit.angstrom**3, app.element.hydrogen: 0.5e-3}},
    ], ids=['default', 'epsilon', 'thole', 'polarizabilities'])
    def testEnergy(self, platform_int, options):
        """The electrostatic embedding adds the Coulomb correction and induction energy
        to the mechanical embedding, and leaves the MM forces unchanged."""
        prmtop, inpcrd, mmSystem, mlAtoms = _loadTolueneExplicit()
        potential = MLPotential('aceff-2-resp-1')
        meSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, forceGroup=1)
        eeSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, forceGroup=1,
                                               embedding='electrostatic', **options)
        platform = mm.Platform.getPlatform(platform_int)
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), platform)
        meContext = mm.Context(meSystem, mm.VerletIntegrator(0.001), platform)
        eeContext = mm.Context(eeSystem, mm.VerletIntegrator(0.001), platform)
        for context in (mmContext, meContext, eeContext):
            context.setPositions(inpcrd.positions)
            context.setPeriodicBoxVectors(*inpcrd.boxVectors)

        # The MM part is the same as for the mechanical embedding.
        assert np.isclose(_groupEnergy(meContext, 0), _groupEnergy(eeContext, 0), rtol=1e-6)

        # Reference for the embedding terms, using the PME parameters OpenMM actually uses.
        nonbonded = [f for f in mmSystem.getForces() if isinstance(f, mm.NonbondedForce)][0]
        # Read alpha from a Reference context: getPMEParametersInContext() can raise
        # std::bad_cast on the CPU platform.  OpenMM picks alpha the same way on all platforms.
        alpha = nonbonded.getPMEParametersInContext(mm.Context(mmSystem, mm.VerletIntegrator(0.001), mm.Platform.getPlatform('Reference')))[0]
        cutoff = nonbonded.getCutoffDistance().value_in_unit(unit.nanometer)
        charges = np.array([nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                            for i in range(mmSystem.getNumParticles())])
        positions = inpcrd.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = np.array([v.value_in_unit(unit.nanometer) for v in inpcrd.boxVectors])
        atoms = list(prmtop.topology.atoms())
        pols = dict(REFERENCE_POLARIZABILITIES)
        if 'polarizabilities' in options:
            pols = {'C': 1.5e-3, 'H': 0.5e-3}
        polarizabilities = np.array([pols[atoms[i].element.symbol] for i in mlAtoms])
        expected = _referenceEmbeddingEnergy(positions, box, mlAtoms, REFERENCE_RESP_CHARGES, charges, alpha, cutoff, polarizabilities,
                                             options.get('epsilon', 2.0), options.get('tholeDamping', None))

        embeddingEnergy = _groupEnergy(eeContext, 1) - _groupEnergy(meContext, 1)
        assert np.isclose(embeddingEnergy, expected, rtol=1e-3, atol=0.05)

    def testForces(self, platform_int):
        """Forces of the ML group (which include the dependence of the charges on the ML
        positions and forces on the MM atoms) match a finite difference of its energy."""
        prmtop, inpcrd, mmSystem, mlAtoms = _loadTolueneExplicit()
        potential = MLPotential('aceff-2-resp-1')
        eeSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, forceGroup=1,
                                               embedding='electrostatic', tholeDamping=1.3)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(eeSystem, mm.VerletIntegrator(0.001), platform)
        context.setPeriodicBoxVectors(*inpcrd.boxVectors)
        positions = inpcrd.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        context.setPositions(positions)
        forces = context.getState(getForces=True, groups={1}).getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)

        # The MM atoms feel the field of the ML atoms.
        mmAtoms = [i for i in range(len(positions)) if i not in set(mlAtoms)]
        assert np.abs(forces[mmAtoms]).max() > 0

        # Displace all atoms along the force and compare with the change in energy.
        step = 1e-3
        norm = np.linalg.norm(forces)
        direction = forces/norm
        context.setPositions(positions + step*direction)
        e1 = _groupEnergy(context, 1)
        context.setPositions(positions - step*direction)
        e2 = _groupEnergy(context, 1)
        assert np.isclose((e2 - e1)/(2*step), norm, rtol=1e-2)


def testElectrostaticEmbeddingRequiresPME():
    prmtop, _, mmSystem, mlAtoms = _loadTolueneExplicit(nonbondedMethod=app.CutoffPeriodic)
    potential = MLPotential('aceff-2-resp-1')
    with pytest.raises(ValueError, match='PME'):
        potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, embedding='electrostatic')

def testElectrostaticEmbeddingIsSupported():
    assert 'electrostatic' in MLPotential('aceff-2-resp-1').getSupportedEmbeddings()
