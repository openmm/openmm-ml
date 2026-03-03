"""
fennixpotential.py: Support for FeNNix potentials.

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

class FeNNixPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates FeNNixPotentialImpl objects."""

    def createImpl(self, name: str, modelPath: str | None = None, **args) -> MLPotentialImpl:
        return FeNNixPotentialImpl(name, modelPath)

class FeNNixPotentialImpl(MLPotentialImpl):
    """
    Implementation of FeNNix potentials for OpenMM.

    The FeNNol library is used to load FeNNix models and evaluate energies and
    forces.  This implementation can use local files for models or automatically
    download them from the FeNNol-PMC repository.

    To use one of the pre-trained FeNNix models, specify it by name.  For example:

    >>> potential = MLPotential('fennix-bio1S')

    Other available models include 'fennix-bio1M', 'fennix-bio1S-finetuneIons', and 'fennix-bio1M-finetuneIons'.

    To use a local `.fnx` file, specify 'fennix' as the model name, and supply the `modelPath` argument, *e.g.*,

    >>> potential = MLPotential('fennix', modelPath='custom_fennix_model.fnx')
    """

    KNOWN_MODELS = {
        "fennix-bio1S": ("https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/FENNIX-BIO1/v1.0/fennix-bio1S.fnx", True),
        "fennix-bio1M": ("https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/FENNIX-BIO1/v1.0/fennix-bio1M.fnx", True),
        "fennix-bio1S-finetuneIons": ("https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/FENNIX-BIO1/v1.0-finetuneIons/fennix-bio1S-finetuneIons.fnx", True),
        "fennix-bio1M-finetuneIons": ("https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/FENNIX-BIO1/v1.0-finetuneIons/fennix-bio1M-finetuneIons.fnx", True),
    }

    def __init__(self, name: str, modelPath: str | None) -> None:
        """
        Initialize the `FeNNixPotentialImpl`.

        Parameters
        ----------
        name : str
            The name of the model.  Options include the pre-trained models
            'fennix-bio1S', 'fennix-bio1M', 'fennix-bio1S-finetuneIons', and
            'fennix-bio1M-finetuneIons', or 'fennix' to load a local model file.
        modelPath : str, optional
            A path to the model file to load.
        """

        self.name = name
        self.modelPath = modelPath

    def addForces(self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Iterable[int] | None,
        forceGroup: int,
        charge: int = 0,
        precision: str = "single",
        **args
    ) -> None:

        try:
            import fennol
        except ImportError:
            raise ImportError("Failed to import FeNNol: for installation instructions, visit https://github.com/FeNNol-tools/FeNNol")
        import jax.numpy as jnp
        import numpy as np

        # Check precision argument.
        if precision == "single":
            useDouble = False
        elif precision == "double":
            useDouble = True
        else:
            raise ValueError(f"Invalid precision {precision} (expected single or double)")

        # Download or look up the model file to use.
        if self.name in FeNNixPotentialImpl.KNOWN_MODELS:
            url, warn = FeNNixPotentialImpl.KNOWN_MODELS[self.name]
            if warn:
                import logging
                logging.warning(f"The model {self.name} is distributed under the restrictive ASL license.  Commercial use is not permitted.")
            modelPath = self._downloadOrFindFile(f"{self.name}.fnx", url)
        elif self.name == "fennix":
            if self.modelPath is None:
                raise ValueError("No modelPath provided for local FeNNix model.")
            modelPath = self.modelPath
        else:
            raise ValueError(f"Unsupported FeNNix model: {self.name} (options are {", ".join(FeNNixPotentialImpl.KNOWN_MODELS)})")

        # Load the model.
        model = fennol.FENNIX.load(modelPath, **args)
        energyScale = (unit.hartree / model.Ha_to_model_energy * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        forceScale = (energyScale * unit.angstrom).value_in_unit(unit.nanometer)

        # Get the atoms that should be included.
        includedAtoms = list(topology.atoms())
        indices = None
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
            indices = np.array(atoms, dtype=int)

        # Prepare inputs to the model that remain constant from step to step.
        species = jnp.array([atom.element.atomic_number for atom in includedAtoms], dtype=jnp.int32)
        inputs = dict(
            species=species,
            natoms=jnp.array([species.size], dtype=jnp.int32),
            batch_index=jnp.zeros(species.size, dtype=jnp.int32),
            total_charge=charge,
        )

        # Create the PythonForce and add it to the System.
        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()
        force = openmm.PythonForce(_ComputeFeNNix(model, energyScale, forceScale, indices, inputs, periodic, useDouble))
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)

class _ComputeFeNNix:
    def __init__(self, model, energyScale, forceScale, indices, inputs, periodic, useDouble):
        self.model = model
        self.energyScale = energyScale
        self.forceScale = forceScale
        self.indices = indices
        self.inputs = inputs
        self.periodic = periodic
        self.useDouble = useDouble

    def __call__(self, state):
        import jax
        import numpy as np

        # Load coordinates and box vectors from the state.
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        numAtoms = positions.shape[0]
        if self.indices is not None:
            positions = positions[self.indices]
        if self.periodic:
            cells = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom).reshape(1, 3, 3)

        # Invoke the model to get the energy and forces.
        with jax.enable_x64(self.useDouble):
            if self.periodic:
                modelOutputs = self.model.energy_and_forces(coordinates=positions, cells=cells, **self.inputs)
            else:
                modelOutputs = self.model.energy_and_forces(coordinates=positions, **self.inputs)
            jaxEnergy, jaxForces = modelOutputs[:2]
            energy = jaxEnergy.item() * self.energyScale
            jaxForces *= self.forceScale
            if self.indices is None:
                forces = np.asarray(jaxForces)
            else:
                forces = np.zeros((numAtoms, 3), dtype=jaxForces.dtype)
                forces[self.indices] = jaxForces

        return energy, forces

    def __getstate__(self):
        return (self.model.to_dict(), self.energyScale, self.forceScale, self.indices, self.inputs, self.periodic, self.useDouble)

    def __setstate__(self, pickle_state):
        import fennol
        model_dict, self.energyScale, self.forceScale, self.indices, self.inputs, self.periodic, self.useDouble = pickle_state
        self.model = fennol.FENNIX(**model_dict)
