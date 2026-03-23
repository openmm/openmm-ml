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

    def createImpl(self, name: str, modelName: str | None = None, **args) -> MLPotentialImpl:
        return OrbPotentialImpl(name, modelName)

class OrbPotentialImpl(MLPotentialImpl):
    """
    Implementation of Orb potentials for OpenMM.

    To use one of the latest recommended Orb models, specify it by name:

    >>> potential = MLPotential('orb-v3-omol')

    Any Orb model available with the orb-models package, including older models
    and those not recommended for standard use, can be loaded by specifying
    'orb' as the model, and its full name with the `modelName` argument.

    >>> potential = MLPotential('orb', modelName='orb-v3-conservative-inf-mpa')
    """

    KNOWN_MODELS = {
        "orb-v3-omat": "orb-v3-conservative-inf-omat",
        "orb-v3-omol": "orb-v3-conservative-omol",
    }

    def __init__(self, name: str, modelName: str | None) -> None:
        """
        Initialize the `OrbPotentialImpl`.

        Parameters
        ----------
        name : str
            An abbreviated model name ('orb-v3-omat' or 'orb-v3-omol'), or
            'orb' to load any model available in the orb-models package by its
            full name.
        modelName : str, optional
            The full name of a model to load if `name` is 'orb'.
        """

        self.name = name
        self.modelName = modelName

    def addForces(self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Iterable[int] | None,
        forceGroup: int,
        charge: int = 0,
        multiplicity: int = 1,
        precision: str = "float32-high",
        **args
    ) -> None:

        try:
            from orb_models.forcefield import pretrained as orb
            from orb_models.forcefield.models.conservative_regressor import ConservativeForcefieldRegressor
        except ImportError:
            raise ImportError("Failed to import orb-models: for installation instructions, visit https://github.com/orbital-materials/orb-models")
        import ase
        import numpy as np

        # Check precision argument.
        if precision not in ("float32-high", "float32-highest", "float64"):
            raise ValueError(f"Invalid precision {precision} (expected float32-high, float32-highest, or float64)")

        # Resolve the model name options to the name of a known Orb model.
        if self.name in OrbPotentialImpl.KNOWN_MODELS:
            modelName = OrbPotentialImpl.KNOWN_MODELS[self.name]
        elif self.name == "orb":
            if self.modelName is None:
                raise ValueError("No modelName provided for Orb model.")
            if self.modelName not in orb.ORB_PRETRAINED_MODELS:
                supported = ", ".join(list(orb.ORB_PRETRAINED_MODELS))
                raise ValueError(f"Unsupported Orb model name: {self.modelName} (options are {supported})")
            modelName = self.modelName
        else:
            supported = ", ".join(list(OrbPotentialImpl.KNOWN_MODELS) + ["orb"])
            raise ValueError(f"Unsupported Orb model preset: {self.name} (options are {supported})")

        # Load the model.
        device = self._getTorchDevice(args)
        model, adapter = orb.ORB_PRETRAINED_MODELS[modelName](device=device, precision=precision)
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
