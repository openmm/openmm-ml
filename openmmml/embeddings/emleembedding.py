"""
emleembedding.py: Provides the EMLE (Electrostatic Machine Learning Embedding)
method as an embedding plugin.

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

from openmmml.mlpotential import MLPotentialImpl, Embedding, EmbeddingFactory
from openmmml.embeddings import utilities
import openmm
import openmm.app
import openmm.unit as unit
import numpy as np
from functools import partial

class EMLEEmbeddingFactory(EmbeddingFactory):
    """This is the factory that creates EMLEEmbedding objects."""

    def createEmbedding(self, name: str, **args) -> Embedding:
        return EMLEEmbedding(name)


class EMLEEmbedding(Embedding):
    """EMLE (Electrostatic Machine Learning Embedding).  This embedding method
    can be used with any ML potential to perform ML/MM simulations.  The ML-ML
    interactions will be computed with the ML potential of choice, the MM-MM
    interactions with a conventional force field, and the remaining interactions
    with EMLE.
    """

    def __init__(self, name: str):
        """
        Initialize the EMLEEmbedding.

        Parameters
        ----------
        name : str
            The name of the EMLE model.  `emle` (the only builtin model) uses
            the standard EMLE model.  `emle-engine` allows you to use a custom
            model supported by the EMLE library.
        """

        self.name = name

    def createMixedSystem(self,
                          potential: MLPotentialImpl,
                          topology: openmm.app.Topology,
                          system: openmm.System,
                          atoms: list[int],
                          forceGroup: int,
                          interpolate: bool,
                          **args) -> openmm.System:

        # Make sure that we can import the EMLE library.

        try:
            from emle.models import EMLE
        except ImportError:
            raise ImportError("Failed to import emle-engine: for installation instructions, visit https://github.com/chemle/emle-engine")
        import torch

        # Get the model path to pass to EMLE.

        if self.name == "emle":
            modelPath = None
        elif self.name == "emle-engine":
            try:
                modelPath = args["embeddingModelPath"]
            except KeyError:
                raise ValueError("For the emle-engine embedding method, an embeddingModelPath must be provided")
        else:
            raise ValueError(f"Unrecognized embedding name {self.name!r} for EMLE (recognized names are emle, emle-engine)")

        # Extract additional options to pass to EMLE.

        precision = args.get("precision", None)
        alphaMode = args.get("alphaMode", "species")

        # Create the new system with ML-ML interactions to be computed by the ML
        # potential removed.

        periodic = system.usesPeriodicBoundaryConditions()
        newSystem = utilities.removeBonds(system, atoms, True)
        numAtoms = newSystem.getNumParticles()

        allCharges = [0.0] * numAtoms
        for force in newSystem.getForces():
            if isinstance(force, openmm.NonbondedForce):
                # Get charges on all particles.

                for atom in range(numAtoms):
                    charge, _, _ = force.getParticleParameters(atom)
                    allCharges[atom] += charge.value_in_unit(unit.elementary_charge)

                # All of the LJ interactions in the ML region should be zeroed.
                # The ML-ML and ML-MM electrostatics should both be zeroed.

                for atom in atoms:
                    _, sigma, epsilon = force.getParticleParameters(atom)
                    force.setParticleParameters(atom, 0, sigma, epsilon)

                for iAtom1 in range(len(atoms)):
                    for iAtom2 in range(iAtom1):
                        force.addException(atoms[iAtom1], atoms[iAtom2], 0, 1, 0, True)

                # This may cause exceptions in the MM region to use PBCs, but
                # this should not ordinarily have any significant effects.

                force.setExceptionsUsePeriodicBoundaryConditions(periodic)

            elif isinstance(force, openmm.CustomNonbondedForce):
                utilities.makeCustomNonbondedExclusions(force, atoms)

        # Create a PythonForce to compute the EMLE interaction.

        device = potential._getTorchDevice(args)

        if precision is None:
            # This is the default used by the EMLE library if None is given.
            dtype = torch.get_default_dtype()
        elif precision == "single":
            dtype = torch.float32
        elif precision == "double":
            dtype = torch.float64
        else:
            raise ValueError(f"Unsupported precision {precision} for the embedding. Supported values are 'single' and 'double'.")

        mlAtomSet = set(atoms)
        mmAtomList = sorted(set(range(numAtoms)) - mlAtomSet)
        topologyAtoms = list(topology.atoms())

        atomicNumbers = torch.tensor([topologyAtoms[atom].element.atomic_number for atom in atoms], device=device, dtype=int)
        mlCharge = sum(allCharges[atom] for atom in atoms)
        mmCharges = torch.tensor([allCharges[atom] for atom in mmAtomList], device=device)
        mlIndices = torch.tensor(atoms, device=device, dtype=int)
        mmIndices = torch.tensor(mmAtomList, device=device, dtype=int)

        # Trying to pass a float charge to EMLE will give an error.

        mlChargeRounded = round(mlCharge)
        if not np.isclose(mlChargeRounded, mlCharge):
            raise ValueError(f"Non-integer charge on the ML region {mlCharge} unsupported by EMLE")

        model = EMLE(model=modelPath, method="electrostatic", alpha_mode=alphaMode, device=device, dtype=dtype)
        energyScale = (1.0 * unit.hartree / unit.item).value_in_unit(unit.kilojoule_per_mole)
        emleForce = openmm.PythonForce(partial(
            _computeEMLE,
            atomicNumbers=atomicNumbers,
            mlCharge=mlChargeRounded,
            mmCharges=mmCharges,
            mlIndices=mlIndices,
            mmIndices=mmIndices,
            periodic=periodic,
            device=device,
            dtype=dtype,
            model=model,
            energyScale=energyScale,
        ))
        emleForce.setForceGroup(forceGroup)
        emleForce.setUsesPeriodicBoundaryConditions(periodic)

        if interpolate:
            interpolator = utilities.InterpolationHelper()
            interpolator.addMLPotentialTerms(potential, topology, atoms, forceGroup, **args)
            interpolator.addMLTerm(emleForce)
            interpolator.addMMBondedTerms(system, atoms)
            interpolator.setupNonbonded(newSystem, system)
            interpolator.setupInterpolation(newSystem)

        else:
            potential.addForces(topology, newSystem, atoms, forceGroup, **args)
            newSystem.addForce(emleForce)

        return newSystem

def _computeEMLE(state, atomicNumbers, mlCharge, mmCharges, mlIndices, mmIndices, periodic, device, dtype, model, energyScale):
    import torch

    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    positionsTensor = torch.tensor(positions, dtype=dtype, device=device, requires_grad=True)

    if periodic:
        cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
        cellTensor = torch.tensor(cell, dtype=dtype, device=device)
    else:
        cellTensor = None

    energy = energyScale * model(atomicNumbers, mmCharges, positionsTensor[mlIndices], positionsTensor[mmIndices], cellTensor, mlCharge)
    energy = energy.sum()
    # For unknown reasons, retain_graph=True appears necessary when calling EMLE
    # even though we are only calling backward() one time.
    energy.backward(retain_graph=True)
    return energy.item(), (-positionsTensor.grad).numpy(force=True)
