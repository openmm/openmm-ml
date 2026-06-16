"""
mechanicalembedding.py: Implements mechanical embedding.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.
Authors: Peter Eastman, Evan Pretti
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

from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLEmbeddingImpl, MLEmbeddingImplFactory
import openmm
import openmm.app
import openmm.unit as unit
from copy import deepcopy

class MechanicalEmbeddingImplFactory(MLEmbeddingImplFactory):
    """This is the factory that creates MechanicalEmbeddingImpl objects."""

    def createImpl(self, name: str, **args) -> MLEmbeddingImpl:

        # name should always be "mechanical" for this plugin.

        return MechanicalEmbeddingImpl()


class MechanicalEmbeddingImpl(MLEmbeddingImpl):
    """Mechanical embedding.  This is the default embedding method, and uses the
    conventional force field to compute the interactions between the atoms
    within the ML subset and those outside of it.

    If long-range electrostatics are in use by the conventional force field but
    not supported by the ML model, the conventional force field charges will be
    used to approximate the interactions between atoms within the ML subset in
    different periodic images.  For some models, this cannot be determined based
    on the information provided to OpenMM-ML, and the mlLongRange option must be
    provided explicitly to control whether or not to include these interactions.
    """

    def __init__(self):
        pass

    def createMixedSystem(self,
                          potential: MLPotentialImpl,
                          topology: openmm.app.Topology,
                          system: openmm.System,
                          atoms: list[int],
                          forceGroup: int,
                          interpolate: bool,
                          **args) -> openmm.System:

        periodic = system.usesPeriodicBoundaryConditions()

        # See if the ML potential uses long-range interactions.  mlLongRange may
        # end up being None, in which case we were unable to determine this.

        potentialMLLongRange = potential.getMLLongRange()
        userMLLongRange = args.get("mlLongRange", None)
        if potentialMLLongRange is None:
            mlLongRange = userMLLongRange
        else:
            if userMLLongRange is not None:
                raise ValueError("This ML model does not support the mlLongRange option.")
            mlLongRange = potentialMLLongRange

        # See if the MM force field uses long-range interactions.

        mmLongRange = False
        mmLongRangeForce = None
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                if force.getNonbondedMethod() in (openmm.NonbondedForce.Ewald, openmm.NonbondedForce.PME, openmm.NonbondedForce.LJPME):
                    mmLongRange = True
                    if mmLongRangeForce is not None:
                        raise ValueError("Multiple long-range NonbondedForce objects encountered.")
                    mmLongRangeForce = force
                    break

        excludeLongRange = False

        if periodic:

            # For a non-periodic system, we will always use exceptions for the
            # ML subset.  Otherwise, we need to know whether or not the ML
            # potential is long-range.

            if mlLongRange is None:
                raise ValueError("The system is periodic and it is unknown if the ML model uses long-range interactions; provide the mlLongRange option to specify.")

            # We don't support the case where the MM force field is not
            # long-range but the ML potential is.

            if mlLongRange:
                if not mmLongRange:
                    raise ValueError("The system is periodic and the ML model uses long-range interactions while the MM force field does not.")

                excludeLongRange = True

        # Create the new system with ML-ML interactions to be computed by the ML
        # potential removed.

        newSystem = MLPotential._removeBonds(system, atoms, True)

        for force in newSystem.getForces():
            if isinstance(force, openmm.NonbondedForce):
                # All of the LJ interactions in the ML region should be zeroed.
                charges = [force.getParticleParameters(atom)[0] for atom in atoms]
                for iAtom1 in range(len(atoms)):
                    for iAtom2 in range(iAtom1):

                        # If the ML region electrostatics should be excluded in
                        # long-range, keep this exception's charge product set
                        # to the true product of the charges since the energy
                        # for this pair will be subtracted later.  If not, set
                        # it to zero here.

                        if excludeLongRange:
                            chargeProd = charges[iAtom1] * charges[iAtom2]
                        else:
                            chargeProd = 0
                        force.addException(atoms[iAtom1], atoms[iAtom2], chargeProd, 1, 0, True)

                # This may cause exceptions in the MM region to use PBCs, but
                # this should not ordinarily have any significant effects.

                force.setExceptionsUsePeriodicBoundaryConditions(periodic)

            elif isinstance(force, openmm.CustomNonbondedForce):
                existing = set(tuple(force.getExclusionParticles(i)) for i in range(force.getNumExclusions()))
                for iAtom1 in range(len(atoms)):
                    atom1 = atoms[iAtom1]
                    for iAtom2 in range(iAtom1):
                        atom2 = atoms[iAtom2]
                        if (atom1, atom2) not in existing and (atom2, atom1) not in existing:
                            force.addExclusion(atom1, atom2)

        if excludeLongRange:
            # Prepare a force to calculate the PME energy of the ML-ML region.

            excludeForce = openmm.NonbondedForce()
            excludeForce.setCutoffDistance(mmLongRangeForce.getCutoffDistance())
            excludeForce.setEwaldErrorTolerance(mmLongRangeForce.getEwaldErrorTolerance())
            excludeForce.setNonbondedMethod(openmm.NonbondedForce.PME)
            excludeForce.setPMEParameters(*mmLongRangeForce.getPMEParameters())

            atomSet = set(atoms)
            for atom in range(newSystem.getNumParticles()):
                excludeForce.addParticle(mmLongRangeForce.getParticleParameters(atom)[0] if atom in atomSet else 0, 1, 0)

        if interpolate:
            # Set up a CV force to do the interpolation.

            cvForce = openmm.CustomCVForce("")
            cvForce.addGlobalParameter("lambda_interpolate", 1)

            mlTermNames = []
            def addMLTerm(force):
                name = f"mlTerm{len(mlTermNames)}"
                cvForce.addCollectiveVariable(name, force)
                mlTermNames.append(name)

            mmTermNames = []
            def addMMTerm(force):
                name = f"mmTerm{len(mmTermNames)}"
                cvForce.addCollectiveVariable(name, force)
                mmTermNames.append(name)

            # Add the ML potential.

            tempSystem = openmm.System()
            potential.addForces(topology, tempSystem, atoms, forceGroup, **args)
            for force in tempSystem.getForces():
                addMLTerm(deepcopy(force))

            # Add back the bonds that were removed.

            bondedSystem = MLPotential._removeBonds(system, atoms, False)
            for force in bondedSystem.getForces():
                if hasattr(force, "addBond") or hasattr(force, "addAngle") or hasattr(force, "addTorsion"):
                    addMMTerm(deepcopy(force))

            # Add in terms corresponding to the exceptions set earlier.

            for force in system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    replacementForce = openmm.CustomBondForce("138.935456*chargeProd/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)")
                    replacementForce.setUsesPeriodicBoundaryConditions(periodic)
                    replacementForce.addPerBondParameter("chargeProd")
                    replacementForce.addPerBondParameter("sigma")
                    replacementForce.addPerBondParameter("epsilon")

                    atomCharge, atomSigma, atomEpsilon = zip(*(force.getParticleParameters(atom) for atom in atoms))
                    exceptions = {}
                    for i in range(force.getNumExceptions()):
                        atom1, atom2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                        exceptions[atom1, atom2] = (chargeProd, sigma, epsilon)
                    def getParameters(iAtom1, iAtom2):
                        atom1 = atoms[iAtom1]
                        atom2 = atoms[iAtom2]
                        if (atom1, atom2) in exceptions:
                            return exceptions[atom1, atom2]
                        if (atom2, atom1) in exceptions:
                            return exceptions[atom2, atom1]
                        return (
                            atomCharge[iAtom1] * atomCharge[iAtom2],
                            0.5 * (atomSigma[iAtom1] + atomSigma[iAtom2]),
                            unit.sqrt(atomEpsilon[iAtom1] * atomEpsilon[iAtom2]),
                        )

                    for iAtom1 in range(len(atoms)):
                        for iAtom2 in range(iAtom1):
                            chargeProd, sigma, epsilon = getParameters(iAtom1, iAtom2)
                            if excludeLongRange:
                                chargeProd -= atomCharge[iAtom1] * atomCharge[iAtom2]
                            if chargeProd._value != 0 or epsilon._value != 0:
                                replacementForce.addBond(atoms[iAtom1], atoms[iAtom2], [chargeProd, sigma, epsilon])

                    addMMTerm(replacementForce)

                elif isinstance(force, openmm.CustomNonbondedForce):
                    # TODO: this case has never been supported (previously it
                    # would be silently ignored).
                    raise NotImplementedError("Interpolation is currently unsupported when a CustomNonbondedForce is present")

            # Subtract the ML-ML PME energy if needed.

            if excludeLongRange:
                mlTermNames.append("(-excludeForce)")
                cvForce.addCollectiveVariable("excludeForce", excludeForce)

            # Build the expression to do the interpolation.

            cvTerms = []
            if mlTermNames:
                mlSum = "+".join(mlTermNames)
                cvTerms.append(f"lambda_interpolate*({mlSum})")
            if mmTermNames:
                mmSum = "+".join(mmTermNames)
                cvTerms.append(f"(1-lambda_interpolate)*({mmSum})")
            cvForce.setEnergyFunction("+".join(cvTerms) if cvTerms else "0")
            newSystem.addForce(cvForce)

        else:
            # Add the ML potential and subtract the ML-ML PME energy if needed.

            if excludeLongRange:
                cvForce = openmm.CustomCVForce("-excludeForce")
                cvForce.addCollectiveVariable("excludeForce", excludeForce)
                newSystem.addForce(cvForce)

            potential.addForces(topology, newSystem, atoms, forceGroup, **args)

        return newSystem
