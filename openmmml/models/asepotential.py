"""
asepotential.py: Potential functions implemented with an ASE calculator

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.
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

from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
import openmm
from openmm import unit
from typing import Iterable, Optional
from functools import partial

class ASEPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates ASEPotentialImpl objects."""

    def createImpl(self, name: str, **args) -> MLPotentialImpl:
        return ASEPotentialImpl(name)


class ASEPotentialImpl(MLPotentialImpl):
    """This MLPotentialImpl implements potentials using an ASE calculator.

    It can be used in two ways.  One option is to provide an ASE Atoms object that has already been configured with the
    correct calculator and all other settings.

    >>> potential = MLPotential('ase')
    >>> system = potential.createSystem(topology, aseAtoms=atoms)

    When used this way, you must make sure the Atoms object describes exactly the same atoms in the same order as the
    OpenMM Topology.  If it does not, you will get incorrect results.

    Alternatively you can provide a calculator.  In this case, `createSystem()` automatically creates an Atoms object
    based on the Topology and assigns the calculator to it.

    >>> system = potential.createSystem(topology, calculator=calculator)

    When used this way, you can optionally include an `info` argument with properties that should be added to the Atoms
    object's info dict.

    >>> system = potential.createSystem(topology, calculator=calculator, info={'charge':2})
    """

    def __init__(self, name):
        self.name = name

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  **args):
        try:
            import ase
        except ImportError as e:
            raise ImportError("Failed to import ASE.  Install it as described at https://ase-lib.org/install.html.")
        if any(atom.element is None for atom in topology.atoms()):
            raise ValueError('All atoms in the Topology must have elements defined.')
        includedAtoms = list(topology.atoms())
        if atoms is None:
            indices = None
        else:
            includedAtoms = [includedAtoms[i] for i in atoms]
            indices = sorted(atoms)
        if 'aseAtoms' in args:
            # The user provided an Atoms object.

            aseAtoms = args['aseAtoms']
            if len(aseAtoms.numbers) != len(includedAtoms):
                raise ValueError('The ASE Atoms object contains the wrong number of atoms.')
            if any(n != atom.element.atomic_number for n, atom in zip(aseAtoms.numbers, includedAtoms)):
                raise ValueError('The elements specified in the ASE Atoms and OpenMM Topology do not agree.')
        else:
            # Create an Atoms object.

            if 'calculator' not in args:
                raise ValueError('Either an Atoms or a Calculator must be provided')
            numbers = [atom.element.atomic_number for atom in includedAtoms]
            pbc = (topology.getPeriodicBoxVectors() is not None)
            aseAtoms = ase.Atoms(numbers=numbers, pbc=pbc, calculator=args['calculator'])
        if 'info' in args:
            for key, value in args['info'].items():
                aseAtoms.info[key] = value

        # Create the PythonForce and add it to the System.

        compute = partial(_computeASE, atoms=aseAtoms, indices=indices)
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(any(aseAtoms.get_pbc()))
        system.addForce(force)


def _computeASE(state, atoms, indices):
    import ase.units
    import numpy as np
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    numAtoms = positions.shape[0]
    if indices is not None:
        positions = positions[indices]
    atoms.set_positions(positions)
    if any(atoms.get_pbc()):
        atoms.set_cell(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom))
    energy = atoms.get_potential_energy(apply_constraint=False)
    forces = atoms.get_forces(apply_constraint=False)
    if indices is not None:
        f = np.zeros((numAtoms, 3), dtype=np.float32)
        f[indices] = forces
        forces = f
    return energy/(ase.units.kJ/ase.units.mol), forces/(10*ase.units.kJ/ase.units.mol)
