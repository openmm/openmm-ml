"""
mlpotential.py: Provides a common API for creating OpenMM Systems with ML potentials.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021 Stanford University and the Authors.
Authors: Peter Eastman
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

import openmm
import openmm.app
from typing import Dict, Iterable, Optional


class MLPotentialImplFactory(object):
    """Abstract interface for classes that create MLPotentialImpl objects.

    If you are defining a new potential function, you need to create subclasses
    of MLPotentialImpl and MLPotentialImplFactory, and register an instance of
    the factory by calling MLPotential.registerImplFactory().
    """
    
    def createImpl(self, name: str, **args) -> "MLPotentialImpl":
        """Create a MLPotentialImpl that will be used to implement a MLPotential.

        When a MLPotential is created, it invokes this method to create an object
        implementing the requested potential.  Subclasses must implement this method
        to return an instance of the correct MLPotentialImpl subclass.

        Parameters
        ----------
        name: str
            the name of the potential that was specified to the MLPotential constructor
        args:
            any additional keyword arguments that were provided to the MLPotential
            constructor are passed to this method.  This allows subclasses to customize
            their behavior based on extra arguments.

        Returns
        -------
        a MLPotentialImpl that implements the potential
        """
        raise NotImplementedError('Subclasses must implement createImpl()')


class MLPotentialImpl(object):
    """Abstract interface for classes that implement potential functions.

    If you are defining a new potential function, you need to create subclasses
    of MLPotentialImpl and MLPotentialImplFactory.  When a user creates a
    MLPotential and specifies a name for the potential to use, it looks up the
    factory that has been registered for that name and uses it to create a
    MLPotentialImpl of the appropriate subclass.
    """
    
    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  **args):
        """Add Force objects to a System to implement the potential function.

        This is invoked by MLPotential.createSystem().  Subclasses must implement
        it to create the requested potential function.

        Parameters
        ----------
        topology: Topology
            the Topology from which the System is being created
        system: System
            the System that is being created
        atoms: Optional[Iterable[int]]
            the indices of atoms the potential should be applied to, or None if
            it should be applied to the entire System
        forceGroup: int
            the force group that any newly added Forces should be in
        args:
            any additional keyword arguments that were provided to createSystem()
            are passed to this method.  This allows subclasses to customize their
            behavior based on extra arguments.
        """
        raise NotImplementedError('Subclasses must implement addForces()')


class MLPotential(object):
    """A potential function that can be used in simulations.

    To use this class, create a MLPotential, specifying the name of the potential
    function to use.  You can then call createSystem() to create a System object
    for a simulation.  For example,

    >>> potential = MLPotential('ani2x')
    >>> system = potential.createSystem(topology)

    Alternatively, you can use createMixedSystem() to create a System where part is
    modeled with this potential and the rest is modeled with a conventional force
    field.  As an example, suppose the Topology contains three chains.  Chain 0 is
    a protein, chain 1 is a ligand, and chain 2 is solvent.  The following code
    creates a System in which the internal energy of the ligand is computed with
    ANI2x, while everything else (including interactions between the ligand and the
    rest of the System) is computed with Amber14.

    >>> forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    >>> mm_system = forcefield.createSystem(topology)
    >>> chains = list(topology.chains())
    >>> ml_atoms = [atom.index for atom in chains[1].atoms()]
    >>> potential = MLPotential('ani2x')
    >>> ml_system = potential.createMixedSystem(topology, mm_system, ml_atoms)
    """

    _implFactories: Dict[str, MLPotentialImplFactory] = {}
    
    def __init__(self, name: str, **args):
        """Create a MLPotential.

        Parameters
        ----------
        name: str
            the name of the potential function to use.  Built in support is currently
            provided for the following: 'ani1ccx', 'ani2x'.  Others may be added by
            calling MLPotential.registerImplFactory().
        args:
            particular potential functions may define additional arguments that can
            be used to customize them.  See the documentation on the specific
            potential functions for more information.
        """
        self._impl = MLPotential._implFactories[name].createImpl(name, **args)
    
    def createSystem(self, topology: openmm.app.Topology, **args) -> openmm.System:
        """Create a System for running a simulation with this potential function.

        Parameters
        ----------
        topology: Topology
            the Topology for which to create a System
        args:
            particular potential functions may define additional arguments that can
            be used to customize them.  See the documentation on the specific
            potential functions for more information.

        Returns
        -------
        a newly created System object that uses this potential function to model the Topology
        """
        system = openmm.System()
        for atom in topology.atoms():
            if atom.element is None:
                system.addParticle(0)
            else:
                system.addParticle(atom.element.mass)
        self._impl.addForces(topology, system, None, 0, **args)
        return system

    def createMixedSystem(self,
                          topology: openmm.app.Topology,
                          system: openmm.System,
                          atoms: Iterable[int],
                          removeConstraints: bool = True,
                          forceGroup: int = 0,
                          **args) -> openmm.System:
        """Create a System that is partly modeled with this potential and partly
        with a conventional force field.

        To use this method, first create a System that is entirely modeled with the
        conventional force.  Pass it to this method, along with the indices of the
        atoms to model with this potential (the "ML subset").  It returns a new System
        that is identical to the original one except for the following changes.

        1. Removing all bonds, angles, and torsions for which *all* atoms are in the
           ML subset.
        2. For every NonbondedForce and CustomNonbondedForce, adding exceptions/exclusions
           to prevent atoms in the ML subset from interacting with each other.
        3. (Optional) Removing constraints between atoms that are both in the ML subset.
        4. Adding Forces as necessary to compute the internal energy of the ML subset
           with this potential.

        Parameters
        ----------
        topology: Topology
            the Topology for which to create a System
        system: System
            a System that models the Topology with a conventional force field
        atoms: Iterable[int]
            the indices of all atoms whose interactions should be computed with
            this potential
        removeConstraints: bool
            if True, remove constraints between pairs of atoms whose interaction
            will be computed with this potential
        forceGroup: int
            the force group the ML potential's Forces should be placed in
        args:
            particular potential functions may define additional arguments that can
            be used to customize them.  See the documentation on the specific
            potential functions for more information.

        Returns
        -------
        a newly created System object that uses this potential function to model the Topology
        """
        # Create an XML representation of the System.

        import xml.etree.ElementTree as ET
        xml = openmm.XmlSerializer.serialize(system)
        root = ET.fromstring(xml)

        # Remove bonds, angles, and torsions.

        atomSet = set(atoms)
        for bonds in root.findall('./Forces/Force/Bonds'):
            for bond in bonds.findall('Bond'):
                bondAtoms = [int(bond.attrib[p]) for p in ('p1', 'p2')]
                if all(a in atomSet for a in bondAtoms):
                    bonds.remove(bond)
        for angles in root.findall('./Forces/Force/Angles'):
            for angle in angles.findall('Angle'):
                angleAtoms = [int(angle.attrib[p]) for p in ('p1', 'p2', 'p3')]
                if all(a in atomSet for a in angleAtoms):
                    angles.remove(angle)
        for torsions in root.findall('./Forces/Force/Torsions'):
            for torsion in torsions.findall('Torsion'):
                torsionAtoms = [int(torsion.attrib[p]) for p in ('p1', 'p2', 'p3', 'p4')]
                if all(a in atomSet for a in torsionAtoms):
                    torsions.remove(torsion)

        # Optionally remove constraints.

        if removeConstraints:
            for constraints in root.findall('./Constraints'):
                for constraint in constraints.findall('Constraint'):
                    constraintAtoms = [int(constraint.attrib[p]) for p in ('p1', 'p2')]
                    if all(a in atomSet for a in constraintAtoms):
                        constraints.remove(constraint)

        # Create a new System from it.

        newSystem = openmm.XmlSerializer.deserialize(ET.tostring(root, encoding='unicode'))

        # Add nonbonded exceptions and exclusions.

        atomList = list(atoms)
        for force in newSystem.getForces():
            if isinstance(force, openmm.NonbondedForce):
                for i in range(len(atomList)):
                    for j in range(i):
                        force.addException(i, j, 0, 1, 0, True)
            elif isinstance(force, openmm.CustomNonbondedForce):
                existing = set(tuple(force.getExclusionParticles(i)) for i in range(force.getNumExclusions()))
                for i in range(len(atomList)):
                    for j in range(i):
                        if (i, j) not in existing and (j, i) not in existing:
                            force.addExclusion(i, j, True)

        # Add the ML potential.

        self._impl.addForces(topology, newSystem, atomList, forceGroup, **args)
        return newSystem

    @staticmethod
    def registerImplFactory(name: str, factory: MLPotentialImplFactory):
        """Register a new potential function that can be used with MLPotential.

        Parameters
        ----------
        name: str
            the name of the potential function that will be passed to the MLPotential constructor
        factory: MLPotentialImplFactory
            a factory object that will be used to create MLPotentialImpl objects
        """
        MLPotential._implFactories[name] = factory
