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
from typing import Dict


class MLPotentialImplFactory(object):
    
    def createImpl(self, name: str, **args) -> "MLPotentialImpl":
        raise NotImplementedError('Subclasses must implement createImpl()')


class MLPotentialImpl(object):
    
    def addForces(self, topology: openmm.app.Topology, system: openmm.System, **args):
        raise NotImplementedError('Subclasses must implement addForces()')


class MLPotential(object):

    _implFactories: Dict[str, MLPotentialImplFactory] = {}
    
    def __init__(self, name: str, **args):
        self._impl = _implFactories[name].createImpl(name, **args)
    
    def createSystem(self, topology: openmm.app.Topology, **args) -> openmm.System:
        system = openmm.System()
        for atom in topology.atoms():
            if atom.element is None:
                system.addParticle(0)
            else:
                system.addParticle(atom.element.mass)
        return self._impl.addForces(topology, system, **args)

    @staticmethod
    def registerImplFactory(name: str, factory: MLPotentialImplFactory):
        MLPotential._implFactories[name] = factory
