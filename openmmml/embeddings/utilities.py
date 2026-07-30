"""
utilities.py: Helper routines for setting up mixed ML/MM simulations.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.
Authors: Evan Pretti
Contributors:

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

import copy
import openmm
import openmm.app
import openmm.unit as unit
from openmmml.mlpotential import MLPotentialImpl

COVALENT_RADII = [
    0, 32, 46, 120, 94, 77, 75, 71, 63, 64, 67, 140, 125, 112, 104, 110, 102,
    99, 96, 176, 154, 133, 122, 121, 110, 107, 104, 100, 99, 101, 109, 112, 109,
    114, 110, 113, 117, 189, 167, 147, 139, 132, 124, 114, 112, 112, 108, 114,
    123, 128, 126, 126, 123, 132, 131, 209, 176, 162, 147, 158, 157, 156, 155,
    151, 152, 151, 150, 149, 149, 148, 153, 146, 137, 131, 123, 118, 115, 111,
    112, 112, 132, 130, 130, 136, 131, 138, 142, 200, 181, 167, 158, 152, 153,
    154, 155, 149, 149, 151, 151, 148, 150, 156, 158, 145, 141, 134, 129, 127,
    121, 115, 114, 109, 122, 136, 143, 146, 158, 148, 157
] * openmm.unit.picometer
"""
Default covalent radii to use for assigning distances in the link-atom method.
This set is taken from MLIPOps, which chose them to be consistent with the
simple-dftd3 library (https://github.com/dftd3/simple-dftd3).  They are taken
from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197, except that the radii
of metals have been reduced by 10%.
"""

def findLinkBonds(topology: openmm.app.Topology, atoms: list[int]) -> list[tuple[int, int]]:
    """
    Finds bonds in a topology between a subset of atoms and its complement.

    Parameters
    ----------
    topology: Topology
        The Topology to find bonds in.
    atoms: list[int]
        A set of atom indices.

    Returns
    -------
    A list of atom index pairs corresponding to "link bonds", i.e., bonds
    between atoms in the subset and atoms not in the subset.  The first index of
    every pair will correspond to the atom in the subset.
    """

    atomSet = set(atoms)
    linkBonds = []

    for bond in topology.bonds():
        atom1 = bond.atom1.index
        atom2 = bond.atom2.index
        atom1Included = atom1 in atomSet
        atom2Included = atom2 in atomSet
        if atom1Included and not atom2Included:
            linkBonds.append((atom1, atom2))
        if atom2Included and not atom1Included:
            linkBonds.append((atom2, atom1))

    return linkBonds

def addLinkAtomSites(topology: openmm.app.Topology, systems: list[openmm.System], linkBonds: list[tuple[int, int]], linkAtomDistances: list[tuple[int, int, unit.Quantity]]) -> list[int]:
    """
    Adds virtual sites to systems and a topology for the link-atom method.

    Each virtual site represents a hydrogen atom capping a bond spanning the ML
    and MM regions of an ML/MM simulation.  By default, the distance from the
    atom on the ML side of such a bond to the virtual site is calculated based
    on the covalent radii of the ML atom and hydrogen, but this is overridable
    for particular link bonds using `linkAtomDistances`.

    Parameters
    ----------
    systems: list[System]
        The list of Systems to modify in place by adding virtual sites.
    topology: Topology
        The Topology to look up atomic numbers from and modify in place by
        adding virtual sites.
    linkBonds: list[tuple[int, int]]
        A list of bonds to add virtual sites to, in the format returned by
        `findLinkBonds()`.
    linkAtomDistances: list[tuple[int, int, Quantity]]
        A list of link bonds with virtual site distances to set manually.

    Returns
    -------
    A list of indices corresponding to the virtual sites added to the systems.
    """

    linkAtomDistanceTable = {}
    for atom1, atom2, distance in linkAtomDistances:
        linkAtomDistanceTable[min(atom1, atom2), max(atom1, atom2)] = distance

    # Update the topology with virtual sites to be added, and load data from it.

    residues = [atom.residue for atom in topology.atoms()]
    siteIndices = []
    for site, (atom1, _) in enumerate(linkBonds):
        siteIndices.append(topology.addAtom(f"V{site}", openmm.app.element.hydrogen, residues[atom1]).index)
    atomicNumbers = [atom.element.atomic_number for atom in topology.atoms()]

    # Add virtual sites to the systems.

    for atom1, atom2 in linkBonds:
        key = min(atom1, atom2), max(atom1, atom2)
        if key in linkAtomDistanceTable:
            distance = linkAtomDistanceTable[key]
        else:
            distance = COVALENT_RADII[atomicNumbers[atom1]] + COVALENT_RADII[1]

        for system in systems:
            site = openmm.LocalCoordinatesSite([atom1, atom2], [1.0, 0.0], [-1.0, 1.0], [0.0, 0.0], [distance, 0.0, 0.0])
            system.setVirtualSite(system.addParticle(0.0), site)

            for force in system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    force.addParticle(0.0, 0.0, 0.0)

    return siteIndices

def removeBonds(system: openmm.System, atoms: list[int], removeInSet: bool, linkBonds: list[tuple[int, int]] | None = None) -> openmm.System:
    """
    Copy a System, removing all bonded interactions between atoms in (or not in)
    a particular set.

    If `linkBonds` is specified, an expanded set of atoms including those bonded
    to atoms in the set by bonds in `linkBonds` will be used for the angles and
    torsions.  This behavior is appropriate for use with the link-atom method.

    Parameters
    ----------
    system: System
        The System to copy.
    atoms: list[int]
        A set of atom indices.
    removeInSet: bool
        If True, any bonded term connecting atoms in the specified set is
        removed.  If False, any term that does *not* connect atoms in the
        specified set is removed.
    linkBonds: list[tuple[int, int]], optional
        A list of bonds spanning the specified set of atoms and its complement.
        The first index in each pair should be that of an atom in the set, and
        the second likewise out of the set.  This is the format returned by
        `findLinkBonds()`.

    Returns
    -------
    A newly created System object in which the specified bonded interactions
    have been removed.
    """

    atomSet = set(atoms)

    if linkBonds is None:
        expandedAtomSet = atomSet
    else:
        expandedAtomSet = set(atomSet)
        for atom1, atom2 in linkBonds:
            assert atom1 in atomSet
            expandedAtomSet.add(atom2)

    # Create an XML representation of the System.

    import xml.etree.ElementTree as ET
    xml = openmm.XmlSerializer.serialize(system)
    root = ET.fromstring(xml)

    # This function decides whether a bonded interaction should be removed.

    def shouldRemove(termAtoms, checkSet):
        return all(a in checkSet for a in termAtoms) == removeInSet

    # Remove bonds, angles, and torsions.

    for bonds in root.findall('./Forces/Force/Bonds'):
        for bond in bonds.findall('Bond'):
            bondAtoms = [int(bond.attrib[p]) for p in ('p1', 'p2')]
            if shouldRemove(bondAtoms, atomSet):
                bonds.remove(bond)
    for angles in root.findall('./Forces/Force/Angles'):
        for angle in angles.findall('Angle'):
            angleAtoms = [int(angle.attrib[p]) for p in ('p1', 'p2', 'p3')]
            if shouldRemove(angleAtoms, expandedAtomSet):
                angles.remove(angle)
    for torsions in root.findall('./Forces/Force/Torsions'):
        for torsion in torsions.findall('Torsion'):
            torsionLabels =  ('p1', 'p2', 'p3', 'p4') if 'p1' in torsion.attrib else ('a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4')
            torsionAtoms = [int(torsion.attrib[p]) for p in torsionLabels]
            if shouldRemove(torsionAtoms, expandedAtomSet):
                torsions.remove(torsion)

    # Create a new System from it.

    return openmm.XmlSerializer.deserialize(ET.tostring(root, encoding='unicode'))

def addCustomNonbondedExclusions(force: openmm.CustomNonbondedForce, atoms: list[int]) -> None:
    """
    Adds exclusions between all atoms in a set to a CustomNonbondedForce.

    Parameters
    ----------
    force: openmm.CustomNonbondedForce
        The CustomNonbondedForce to modify in place.
    atoms: list[int]
        A set of atom indices.
    """

    # Only call addExclusion for those particle pairs not already excluded.

    existing = set(tuple(force.getExclusionParticles(i)) for i in range(force.getNumExclusions()))
    for iAtom1 in range(len(atoms)):
        atom1 = atoms[iAtom1]
        for iAtom2 in range(iAtom1):
            atom2 = atoms[iAtom2]
            if (atom1, atom2) not in existing and (atom2, atom1) not in existing:
                force.addExclusion(atom1, atom2)

class InterpolationHelper:
    """
    Helper class for interpolation between ML and MM systems.  After adding ML
    and MM terms, this class can add forces to a System allowing these terms to
    be turned on and off based on the value of a global parameter.
    """

    def __init__(self, globalName: str = "lambda_interpolate") -> None:
        """
        Initializes an InterpolationHelper.

        Parameters
        ----------
        globalName: str
            The name of the global parameter to use for interpolation.
        """

        self.globalName = globalName
        self.cvForce = openmm.CustomCVForce("")
        self.cvForce.addGlobalParameter(globalName, 1)
        self.mlTerms = []
        self.mmTerms = []

    def addMLTerm(self, force: openmm.Force, formatString: str = "{}") -> None:
        """
        Adds an ML term, i.e., a term that should be on when the interpolation
        parameter is 1 and off when it is 0.

        Parameters
        ----------
        force: openmm.Force
            The OpenMM Force to add.  This Force must not already belong to an
            OpenMM System.
        formatString: str
            The format string for the term.  Set to, e.g., "-{}" to calculate
            the negative of the term, or any other custom expression value.  A
            collective variable name will be substituted for the format field.
        """

        name = f"mlTerm{len(self.mlTerms)}"
        self.cvForce.addCollectiveVariable(name, force)
        self.mlTerms.append(formatString.format(name))

    def addMMTerm(self, force: openmm.Force, formatString: str = "{}") -> None:
        """
        Adds an MM term, *i.e.*, a term that should be on when the interpolation
        parameter is 0 and off when it is 1.

        Parameters
        ----------
        force: openmm.Force
            The OpenMM Force to add.  This Force must not already belong to an
            OpenMM System.
        formatString: str
            The format string for the term.  Set to, e.g., "-{}" to calculate
            the negative of the term, or any other custom expression value.  A
            collective variable name will be substituted for the format field.
        """

        name = f"mmTerm{len(self.mmTerms)}"
        self.cvForce.addCollectiveVariable(name, force)
        self.mmTerms.append(formatString.format(name))

    def addMLPotentialTerms(self, potential: MLPotentialImpl, topology: openmm.app.Topology, atoms: list[int], forceGroup: int, **args) -> None:
        """
        Helper function to add all forces created by an ML potential as ML
        terms for interpolation.

        Parameters
        ----------
        potential: MLPotentialImpl
            The potential to call addForces() on.
        topology: openmm.app.Topology
            The Topology to use for ML potential setup.
        atoms: list[int]
            The indices of the ML region atoms in the ML/MM system.
        forceGroup: int
            The force group to pass to addForces().
        args: dict
            Any additional arguments to pass to addForces().
        """

        tempSystem = openmm.System()
        potential.addForces(topology, tempSystem, atoms, forceGroup, **args)
        for force in tempSystem.getForces():
            self.addMLTerm(copy.deepcopy(force))

    def addMMBondedTerms(self, mmSystem: openmm.System, atoms: list[int], linkBonds: list[tuple[int, int]] | None = None) -> None:
        """
        Helper function to add all bonded forces removed from the ML region of
        an ML/MM system as MM terms for interpolation.

        Parameters
        ----------
        mmSystem: openmm.System
            A pure MM system containing all (ML and MM region) bonded terms.
            This will not be modified and is only used as a reference for the
            terms to interpolate.
        atoms: list[int]
            The indices of the ML region atoms in the ML/MM system.
        linkBonds: list[tuple[int, int]], optional
            A list of bonds spanning the ML and MM regions.  See `removeBonds()`
            for details.
        """

        bondedSystem = removeBonds(mmSystem, atoms, False, linkBonds)
        for force in bondedSystem.getForces():
            if hasattr(force, "addBond") or hasattr(force, "addAngle") or hasattr(force, "addTorsion"):
                self.addMMTerm(copy.deepcopy(force))

    def setupNonbonded(self, mixedSystem: openmm.System, mmSystem: openmm.System) -> None:
        """
        Sets up interpolation between the nonbonded force configuration in the
        given mixed system and that in the given MM system.

        This method may modify mixedSystem, and it is necessary to call
        setupInterpolation() once all interpolation terms have been added to
        finish setting up mixedSystem.

        Parameters
        ----------
        mixedSystem: openmm.System
            A system containing only MM interactions that should not be
            interpolated.  The system may be modified.
        mmSystem: openmm.System
            A pure MM system containing all (ML and MM region) nonbonded terms.
            This will not be modified and is only used as a reference for the
            terms to interpolate.
        """

        def extractNonbondedForce(system):
            """Extracts a NonbondedForce and its index from a System or returns None if one is not present."""
            nonbondedForceIndexed = None
            for index, force in enumerate(system.getForces()):
                if isinstance(force, openmm.NonbondedForce):
                    if nonbondedForceIndexed is not None:
                        raise NotImplementedError("Interpolation is currently unsupported when multiple NonbondedForce forces are present")
                    nonbondedForceIndexed = (force, index)
                elif isinstance(force, openmm.CustomNonbondedForce):
                    raise NotImplementedError("Interpolation is currently unsupported when a CustomNonbondedForce is present")
            return nonbondedForceIndexed

        def extractParameters(nonbondedForce):
            """Returns a list of [charge, sigma, epsilon] for each particle in a NonbondedForce."""
            return [nonbondedForce.getParticleParameters(i) for i in range(nonbondedForce.getNumParticles())]

        def extractExceptions(nonbondedForce):
            """Returns a dictionary {(i, j): (chargeProd, sigma, epsilon)} of exceptions (i < j) for a NonbondedForce."""
            exceptions = {}
            for exception in range(nonbondedForce.getNumExceptions()):
                i, j, chargeProd, sigma, epsilon = nonbondedForce.getExceptionParameters(exception)
                exceptions[min(i, j), max(i, j)] = (chargeProd, sigma, epsilon)
            return exceptions

        def extractInteraction(parameters, exceptions, i, j):
            """Finds (chargeProd, sigma, epsilon) for a pair given the output of extractParameters() and extractExceptions()."""
            key = (min(i, j), max(i, j))
            if key in exceptions:
                return exceptions[key]
            else:
                iCharge, iSigma, iEpsilon = parameters[i]
                jCharge, jSigma, jEpsilon = parameters[j]
                return (
                    iCharge * jCharge,
                    0.5 * (iSigma + jSigma),
                    unit.sqrt(iEpsilon * jEpsilon),
                )

        mixedNonbondedForceIndexed = extractNonbondedForce(mixedSystem)
        mmNonbondedForceIndexed = extractNonbondedForce(mmSystem)

        if mixedNonbondedForceIndexed is None and mmNonbondedForceIndexed is None:
            # No nonbonded forces; nothing to do.
            return

        if mixedNonbondedForceIndexed is None or mmNonbondedForceIndexed is None:
            # NOTE: this case shouldn't be generated by any embedding for now.
            raise NotImplementedError("Unsupported inconsistent presence of NonbondedForce for interpolation")

        mixedNonbondedForce, mixedNonbondedForceIndex = mixedNonbondedForceIndexed
        mmNonbondedForce, _ = mmNonbondedForceIndexed

        assert mixedNonbondedForce.getNumParticles() == mmNonbondedForce.getNumParticles()

        mixedParameters = extractParameters(mixedNonbondedForce)
        mmParameters = extractParameters(mmNonbondedForce)
        mixedExceptions = extractExceptions(mixedNonbondedForce)
        mmExceptions = extractExceptions(mmNonbondedForce)
        mixedExceptionsSet = set(mixedExceptions)
        mmExceptionsSet = set(mmExceptions)

        if mixedParameters == mmParameters:
            # Regular parameters are identical; only the exceptions might
            # differ.  Use a CustomBondForce to evaluate the difference between
            # the sets of exceptions in the two forces.

            differenceForce = openmm.CustomBondForce("138.935456*chargeProd/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)")
            differenceForce.setUsesPeriodicBoundaryConditions(mixedNonbondedForce.getExceptionsUsePeriodicBoundaryConditions())
            differenceForce.addPerBondParameter("chargeProd")
            differenceForce.addPerBondParameter("sigma")
            differenceForce.addPerBondParameter("epsilon")

            for i, j in sorted(mixedExceptionsSet | mmExceptionsSet):
                mixedInteraction = extractInteraction(mixedParameters, mixedExceptions, i, j)
                mmInteraction = extractInteraction(mmParameters, mmExceptions, i, j)

                if mixedInteraction == mmInteraction:
                    # Exception pair parameters are identical; nothing to do.
                    continue

                mixedChargeProd, mixedSigma, mixedEpsilon = mixedInteraction
                mmChargeProd, mmSigma, mmEpsilon = mmInteraction

                chargeProd = mmChargeProd - mixedChargeProd
                epsilon = mmEpsilon - mixedEpsilon

                # Pick the appropriate sigma for the exception.  If both
                # epsilons are non-zero and the sigmas don't match, give up
                # (this case shouldn't be generated by any embedding for now).
                if mixedEpsilon._value != 0:
                    if mmEpsilon._value != 0 and mixedSigma != mmSigma:
                        raise NotImplementedError("Unsupported Lennard-Jones exceptions for interpolation")
                    sigma = mixedSigma
                else:
                    sigma = mmSigma

                if chargeProd._value != 0 or epsilon._value != 0:
                    differenceForce.addBond(i, j, [chargeProd, sigma, epsilon])

            self.addMMTerm(differenceForce)

        else:
            # Regular parameters differ; currently this can only be supported by
            # adding two NonbondedForce objects and interpolating between them.

            mmInterpolateForce = copy.deepcopy(mmNonbondedForce)
            mixedInterpolateForce = copy.deepcopy(mixedNonbondedForce)

            # Make the sets of exceptions in these forces agree with each other.

            for i, j in sorted(mixedExceptionsSet - mmExceptionsSet):
                mmInterpolateForce.addException(i, j, *extractInteraction(mmParameters, mmExceptions, i, j))
            for i, j in sorted(mmExceptionsSet - mixedExceptionsSet):
                mixedInterpolateForce.addException(i, j, *extractInteraction(mixedParameters, mixedExceptions, i, j))

            self.addMMTerm(mmInterpolateForce)
            self.addMLTerm(mixedInterpolateForce)

            # We need to remove the NonbondedForce object from the mixedSystem
            # since it will now be added (as a CV) by setupInterpolation().

            mixedSystem.removeForce(mixedNonbondedForceIndex)

    def setupInterpolation(self, mixedSystem: openmm.System) -> None:
        """
        Sets up a mixed system to support interpolation.

        Parameters
        ----------
        mixedSystem: openmm.System
            A system containing only MM interactions that should not be
            interpolated.  The system will be modified to create a mixed ML/MM
            system with interpolation enabled.
        """

        cvTerms = []

        if self.mlTerms:
            mlSum = "+".join(f"({term})" for term in self.mlTerms)
            cvTerms.append(f"{self.globalName}*({mlSum})")

        if self.mmTerms:
            mmSum = "+".join(f"({term})" for term in self.mmTerms)
            cvTerms.append(f"(1-{self.globalName})*({mmSum})")

        self.cvForce.setEnergyFunction("+".join(cvTerms) if cvTerms else "0")
        mixedSystem.addForce(self.cvForce)
