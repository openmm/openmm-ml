"""
so3lrpotential.py: Support for SO3LR potentials.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.
Authors: Adil Kabylda
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

import os
from typing import Iterable
import openmm
from openmm import unit
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory

class SO3LRPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates SO3LRPotentialImpl objects."""

    def createImpl(self, name: str, modelPath: str | None = None, kspace: str = "auto",
                   kspaceSmearing: float | unit.Quantity | None = None,
                   kspaceSpacing: float | unit.Quantity | None = None, **args) -> MLPotentialImpl:
        return SO3LRPotentialImpl(name, modelPath, kspace, kspaceSmearing, kspaceSpacing)

class SO3LRPotentialImpl(MLPotentialImpl):
    """
    Implementation of SO3LR potentials for OpenMM.

    The `so3lr` package (https://github.com/general-molecular-simulations/so3lr)
    is used to evaluate energies and forces.  Its pretrained models are bundled
    with the package and selected by name.  For example:

    >>> potential = MLPotential('so3lr-tauri-m-2606')

    The available models are 'so3lr-tauri-s-2606', 'so3lr-tauri-m-2606' and
    'so3lr-tauri-l-2606' (the SO3LR Tauri models, covering H to At), and
    'so3lr', the original model for H, C, N, O, F, P, S and Cl.  A fine-tuned
    model can be loaded by passing the path to its directory as `modelPath`.

    SO3LR has long-range electrostatics and dispersion.  In periodic systems,
    both are computed with PME by default (in real space for the original
    'so3lr' model).  To use real space only:

    >>> potential = MLPotential('so3lr-tauri-m-2606', kspace='none')
    """

    def __init__(self, name: str, modelPath: str | None, kspace: str,
                 kspaceSmearing: float | unit.Quantity | None, kspaceSpacing: float | unit.Quantity | None) -> None:
        """
        Initialize the `SO3LRPotentialImpl`.

        Parameters
        ----------
        name : str
            The name of the model: 'so3lr-tauri-s-2606', 'so3lr-tauri-m-2606',
            'so3lr-tauri-l-2606' or 'so3lr'.
        modelPath : str, optional
            The path to the directory of a fine-tuned model, used instead of
            the bundled model of that name.
        kspace : str, optional
            The method for the electrostatics and dispersion of periodic
            systems: 'pme', 'ewald', 'none' (real space only), or 'auto' (the
            default: 'pme', or 'none' for models of the original 'so3lr').
            Triclinic boxes have the dispersion in real space.
        kspaceSmearing : float or Quantity, optional
            The k-space smearing (nanometers if a float; 0.2 nm if omitted).
        kspaceSpacing : float or Quantity, optional
            The k-space mesh spacing or wavelength (nanometers if a float;
            0.155 nm if omitted).
        """
        if kspace not in ("auto", "none", "pme", "ewald"):
            raise ValueError(f"Invalid kspace {kspace} (expected auto, none, pme or ewald)")
        # An absolute path, so that the force still finds the model after a change of
        # directory (it is loaded at the first force evaluation).
        self.model = name if modelPath is None else os.path.abspath(os.path.expanduser(modelPath))
        self.kspace = kspace
        self.smearing = _angstroms(kspaceSmearing, _KSPACE_SMEARING)
        self.spacing = _angstroms(kspaceSpacing, _KSPACE_SPACING)

    def addForces(self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Iterable[int] | None,
        forceGroup: int,
        charge: int = 0,
        multiplicity: int = 1,
        precision: str = "mixed",
        lrCutoff: float | unit.Quantity | None = None,
        **args
    ) -> None:
        _importSo3lr()
        from so3lr.mlff.nn.observable.dispersion_ref_data import (
            check_atomic_numbers_supported, supported_atomic_numbers_for_model)

        model = self.model
        if model.startswith("so3lr-tauri-"):
            import logging
            logging.warning(f"The model {model} is distributed under the CC BY-NC 4.0 license.  Commercial use is not permitted.")
        if precision not in _PRECISIONS:
            raise ValueError(f"Invalid precision {precision} (expected single, double or mixed)")

        # Get the atoms that should be included.
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        numbers = [atom.element.atomic_number for atom in includedAtoms]
        # The engine checks this too, but only when it is built at the first step.
        check_atomic_numbers_supported(numbers, supported_atomic_numbers_for_model(model), context=f"SO3LR model {model}")
        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()
        engineArgs = dict(model=model, charge=charge, multiplicity=multiplicity, precision=_PRECISIONS[precision])
        method = self._kspaceMethod(periodic)
        if method:
            engineArgs.update(kspace_electrostatics=method, kspace_smearing=self.smearing, kspace_spacing=self.spacing)
            # so3lr has k-space dispersion only in rectangular boxes.  The System
            # is empty when building an interpolated ML/MM system, so the box
            # comes from the Topology if it has one.
            box = topology.getPeriodicBoxVectors()
            if box is None:
                box = system.getDefaultPeriodicBoxVectors()
            if not _isTriclinic(box):
                engineArgs.update(kspace_dispersion=method, kspace_dispersion_smearing=self.smearing,
                                  kspace_dispersion_spacing=self.spacing)
        if lrCutoff is not None:
            engineArgs["lr_cutoff"] = _angstroms(lrCutoff)
        elif "kspace_dispersion" in engineArgs:
            engineArgs["lr_cutoff"] = _PME_CUTOFF
        elif periodic:
            engineArgs["lr_cutoff"] = _REAL_SPACE_CUTOFF
        else:
            engineArgs["lr_cutoff"] = _GAS_PHASE_CUTOFF
        force = openmm.PythonForce(_ComputeSO3LR(numbers, periodic, engineArgs))
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        if atoms is not None:
            force.setParticles(atoms)
        system.addForce(force)

    def getMLLongRange(self) -> bool:
        # Electrostatics with all periodic images only in k-space; the
        # real-space sum stops at the long-range cutoff.
        return self._kspaceMethod(periodic=True) is not None

    def _kspaceMethod(self, periodic):
        """The k-space method of the electrostatics and the dispersion (None: real space)."""
        if not periodic or self.kspace == "none":
            return None
        if self.kspace == "auto":
            _importSo3lr()
            from so3lr.model_config import is_legacy_model
            return None if is_legacy_model(self.model) else "pme"
        return self.kspace


def _importSo3lr():
    try:
        import so3lr
    except ImportError:
        raise ImportError("Failed to import so3lr.  Install it as described at https://so3lr.github.io/.")


# OpenMM-ML precision -> so3lr precision.  'mixed' is a single precision model
# with double precision positions, sums and results.
_PRECISIONS = {"single": "float32", "mixed": "mixed", "double": "float64"}
_ENERGY_SCALE = (1.0 * unit.ev / unit.item).value_in_unit(unit.kilojoules_per_mole)  # eV -> kJ/mol
_FORCE_SCALE = _ENERGY_SCALE * 10.0  # eV/A -> kJ/mol/nm

# The long-range defaults (Angstroms), as so3lr's command line has them, but with the
# finer 1.55 A mesh for the dispersion too.  Non-periodic: 100 A, so3lr's recommendation
# for molecules (the models were trained at 1000 A).
_PME_CUTOFF = 10.0
_REAL_SPACE_CUTOFF = 12.0
_GAS_PHASE_CUTOFF = 100.0
_KSPACE_SMEARING = 2.0
_KSPACE_SPACING = 1.55


def _isTriclinic(box):
    """Whether reduced box vectors (a along x, b in the xy plane) have off-diagonal terms
    (with so3lr's tolerance of 1e-6 A)."""
    return any(abs(box[i][j].value_in_unit(unit.angstrom)) > 1e-6 for i, j in ((1, 0), (2, 0), (2, 1)))


def _angstroms(value, default=None):
    """A length in Angstroms (so3lr's unit), from a Quantity or a float in nanometers (default if None)."""
    if value is None:
        return default
    return (value if unit.is_quantity(value) else value * unit.nanometers).value_in_unit(unit.angstrom)


class _ComputeSO3LR:
    """The PythonForce computation, with so3lr's So3lrEngine.  The engine is
    built at the first call, so that JAX starts after OpenMM has created its
    Context, and this object is pickled (e.g. by XmlSerializer) as its
    arguments."""

    def __init__(self, numbers, periodic, engineArgs):
        self.numbers = numbers
        self.periodic = periodic
        self.engineArgs = engineArgs
        self.engine = None

    def __reduce__(self):
        return (_ComputeSO3LR, (self.numbers, self.periodic, self.engineArgs))

    def __call__(self, state):
        import jax

        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom) if self.periodic else None
        # 'mixed' and 'double' need JAX's double precision types.
        with jax.enable_x64(self.engineArgs["precision"] != "float32"):
            if self.engine is None:
                from so3lr import So3lrEngine
                self.engine = So3lrEngine(self.numbers, **self.engineArgs)
            energy, forces = self.engine(positions, box)
        return energy * _ENERGY_SCALE, forces * _FORCE_SCALE
