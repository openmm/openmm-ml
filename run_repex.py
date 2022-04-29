#!/usr/bin/env python
"""
demonstration of how to run a MM potential -> NNP/MM hybrid potential free energy calculation with Replica Exchange
TODO : extract free energy and wrap this to perform complex/solute phase for a free energy correction
"""
import torch
import torchani
from NNPOps import OptimizedTorchANI

from openmmtools.testsystems import HostGuestExplicit
from openmmml.mlpotential import MLPotential
from simtk import openmm, unit
import time
import numpy as np
from simtk.openmm import LangevinMiddleIntegrator
from openmmtools.alchemy import AlchemicalState
from openmmtools.states import ThermodynamicState, CompoundThermodynamicState, SamplerState
from copy import deepcopy
from openmmtools import cache
from openmmtools.multistate import replicaexchange
from openmmtools.mcmc import LangevinSplittingDynamicsMove

DEFAULT_CACHE = cache.global_context_cache

class LangevinMiddleDynamicsMove(LangevinSplittingDynamicsMove):
    """
    integrate the `openmm.LangevinMiddleIntegrator` into an mcmc move
    """
    def __init__(self, 
                 n_steps, 
                 timestep = 1.*unti.femtosecond, 
                 collision_rate=10. / unit.picoseconds,
                 reassign_velocities=False,
                 constraint_tolerance = 1e-8,
                 **kwargs
                 )
        super().__init__(timestep=timestep, collision_rate=collision_rate, n_steps=n_steps, reassign_velocities=reassign_velocities,
                         splitting="V R O R V", constraint_tolerance=constraint_tolerance, measure_shadow_work=False, measure_heat=False, **kwargs)

        def _get_integrator(self, thermodynamic_state):
            integrator = LangevinMiddleIntegrator(thermodynamic_state.temperature, self.collision_rate, self.timestep)
            integrator.setConstraintTolerance(self.constraint_tolerance)

class NNPProtocol():
    """
    protocol for perturbing the `scale` parameter of an openmm-ml mixed system
    """
    default_functions = {'scale' : lambda x:x}
    def __init__(self, **kwargs)
        self.functions = deepcopy(self.default_functions)

class NNPAlchemicalState(AlchemicalState):
    """
    neural network potential flavor of `AlchemicalState` for perturbing the `scale` value
    """
    class _LambdaParameter(AlchemicalState._LambdaParameter):
        pass
    
    scale = _LambdaParameter('scale')

    def set_alchemical_parameters(self, global_lambda, lambda_protocol = NNPProtocol()):
        self.global_lambda = global_lambda
        for parameter_name in lambda_protocol.functions:
            lambda_value = lambda_protocol.functions[parameter_name](global_lambda)
            setattr(self, parameter_name, lambda_value)

def minimize_util(thermodynamic_state, sampler_state):
    """
    simple minimization tool
    """
    if type(DEFAULT_CACHE) == cache.DummyContextCache:
        integrator = openmm.VerletInegrator(1.)
        context, integrator = DEFAULT_CACHE.get_context(thermodynamic_state, integrator)
    else:
        context, integrator = cache.global_context_cache.get_context(thermodynamic_state)
    sampler_state.apply_to_context(context, ignore_velocities=True)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=max_iterations)
    sampler_state.update_from_context(context)
    
    

class NNPCompatibilityMixin():
    """
    Mixin that allows the Repex Sampler accommodate NNP protocols
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setup(self, n_states, mixed_system, init_positions, temperature, storage_file, minimisation_steps=1000, n_replicas=None, lambda_schedulg=None, lambda_protocol=NNPProtocol()):
        lambda_zero_alchemical_state = NNPAlchemicalState.from_system(mixed_system)
        thermostate = ThermodynamicState(mixed_system, temperature=temperature)
        compound_thermostate = CompoundThermodynamicState(thermostate, composable_states=[lambda_zero_alchemical_state])
        thermostate_list, sampler_state_list = [], []
        if n_replicas is None:
            n_replicas = n_states
        else:
            raise NotImplementedError(f"the number of states was given as {n_states} but the number of replicas was given as {n_replicas}. We currently only support equal states and replicas")
        if lambda_scedule is None:
            lambda_schedulg = np.linspace(0., 1., n_states)
        else:
            assert len(lambda_scedule) == n_states
            assert np.isclose(lambda_schedule[0], 0.)
            assert np.isclose(lambda_schedule[-1], 1.)
        
        init_sampler_state = SamplerState(init_positions, box_vectors = mixed_system.getDefaultPeriodicBoxVectors())
        for lambda_val in lambda_schedule:
            compound_thermostate_copy = deepcopy(compound_thermostate)
            compound_thermostate_copy.set_alchemical_parameters(lambda_val, lambda_protocol)
            thermostate_list.append(compound_thermostate_copy)
            minimize_util(compound_thermostate_copy, init_sampler_state, max_iterations=minimisation_steps)
            sampler_state_list.append(deepcopy(init_sampler_state))
        
        reporter = storage_file
        
        self.create(thermodynamic_states = thermostate_list, sampler_states = sampler_state_list, storage=reporter)

class NNPRepexSampler(NNPCompatibilityMixin, replicaexchange.ReplicaExchangeSampler):
    """mix the `ReplicaExchangeSampler` with the the mixin"""
    def __init__(self, *args, **kwargs):
        super(NNPRepexSampler).__init__(*args, **kwargs)

"""
# Implementation

temperature = 298.15 * unit.kelvin
#frictionCoeff = 1. / unit.picosecond
#stepSize = 1. * unit.femtoseconds


hgv = HostGuestExplicit(constraints=None)
potential = MLPotential('ani2x')
system = potential.createMixedSystem(hgv.topology, system = hgv.system, atoms = list(range(126,156)), implementation='nnpops', interpolate=True)

sampler = NNPRepexSampler(mcmc_moves = LangevinMiddleDynamicsMove(n_steps=500, 
                                                                  timestep = 1.*unti.femtosecond, 
                                                                  collision_rate=1. / unit.picoseconds,
                                                                  reassign_velocities=True,
                                                                  constraint_tolerance = 1e-6,
                                                                  n_restart_attempts=20),
                         online_analysis_interval=10)

sampler.setup(n_states=13, mixed_system=system, init_positions=hgv.positions, temperature=temperature, storage_file=f"test.nc")  

from openmmtools.utils import get_fastest_platform
platform = get_fastest_platform(minimum_precision='mixed')
platform.setPropertyDefaultValue('DeterministicForces', 'true') # only for CUDA
energy_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)
sampler_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)
sampler.energy_context_cache = energy_context_cache
sampler.sampler_context_cache = sampler_context_cache

# now production
sampler.minimize()
sampler.equilibrate(1000)
sampler.extend(20)
"""
