"""
orbpotential.py: Support for Orb potentials.

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

from functools import partial
from typing import Iterable
import openmm
from openmm import unit
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory

class OrbPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates OrbPotentialImpl objects."""

    def createImpl(self, name: str, **args) -> MLPotentialImpl:
        return OrbPotentialImpl(name)

class OrbPotentialImpl(MLPotentialImpl):
    """
    Implementation of Orb potentials for OpenMM.

    To use one of the recommended pretrained models, specify it by name:

    >>> potential = MLPotential('orb-v3-conservative-omol')

    This gives Orb-v3 (with conservative forces) trained on OMol25.  Orb-v3
    trained on OMat24 is also available as `orb-v3-conservative-inf-omat`.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the `OrbPotentialImpl`.

        Parameters
        ----------
        name : str
            The full name of the Orb model to load.
        """

        self.name = name

    def addForces(self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Iterable[int] | None,
        forceGroup: int,
        charge: int = 0,
        multiplicity: int = 1,
        **args
    ) -> None:

        try:
            from orb_models.forcefield import pretrained as orb
            from orb_models.forcefield.models.conservative_regressor import ConservativeForcefieldRegressor
        except ImportError:
            raise ImportError("Failed to import orb-models: for installation instructions, visit https://github.com/orbital-materials/orb-models")
        try:
            import ase
        except ImportError:
            raise ImportError("Failed to import ASE.  Install it as described at https://ase-lib.org/install.html.")
        import numpy as np

        # Check arguments and load the model.
        if self.name not in orb.ORB_PRETRAINED_MODELS:
            raise ValueError(f"Unsupported Orb model: {self.name}")
        device = self._getTorchDevice(args)
        model, adapter = orb.ORB_PRETRAINED_MODELS[self.name](device=device, precision="float32-highest")
        conservative = isinstance(model, ConservativeForcefieldRegressor)

        # Get the atoms that should be included.
        includedAtoms = list(topology.atoms())
        indices = None
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
            indices = np.array(atoms, dtype=int)

        # Set up the ASE Atoms object that will be fed to the model.
        numbers = [atom.element.atomic_number for atom in includedAtoms]
        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()
        aseAtoms = ase.Atoms(numbers=numbers, pbc=periodic)
        aseAtoms.info['charge'] = charge
        aseAtoms.info['spin'] = multiplicity

        compute = partial(_computeOrb, atoms=aseAtoms, indices=indices, periodic=periodic, device=device, model=model, adapter=adapter, conservative=conservative)
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(any(aseAtoms.get_pbc()))
        system.addForce(force)

def _computeOrb(state, atoms, indices, periodic, device, model, adapter, conservative):
    import ase.units
    import numpy as np

    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    numAtoms = positions.shape[0]
    if indices is not None:
        positions = positions[indices]
    atoms.set_positions(positions)
    if periodic:
        atoms.set_cell(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom))

    result = model.predict(adapter.from_ase_atoms(atoms, device=device))
    energy = result["energy"].item()
    forces = result[model.grad_forces_name if conservative else "forces"].numpy(force=True)
    if indices is not None:
        f = np.zeros((numAtoms, 3), dtype=forces.dtype)
        f[indices] = forces
        forces = f
    return energy / (ase.units.kJ / ase.units.mol), forces / (ase.units.kJ / (ase.units.mol * ase.units.nm))
