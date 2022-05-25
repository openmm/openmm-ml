import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmml.samplers import NNPRepexSampler, LangevinMiddleDynamicsMove
import numpy as np
import pytest
temperature = 298.15 * unit.kelvin


@pytest.mark.parametrize("implementation,platform_name", [
        ('nnpops', 'CUDA'),
        ]) 
class TestSamplers:
    """
    test the `samplers.py` functionality to assert that we can execute a series of replica exchange iterations with a linearly-interpolated MLPotential equipped and 3 states
    """
    def testNNPRepexSampler(self, implementation, platform_name, num_iterations=10, clean=True):
        import torch
        import torchani
        from openmmtools.testsystems import HostGuestExplicit
        from openmmml.mlpotential import MLPotential

        hgv = HostGuestExplicit(constraints=None)
        potential = MLPotential('ani2x')
        system = potential.createMixedSystem(hgv.topology, system = hgv.system, atoms = list(range(126,156)), implementation='nnpops', interpolate=True)
        sampler = NNPRepexSampler(mcmc_moves = LangevinMiddleDynamicsMove(n_steps=500, 
                                                                          timestep = 1.*unit.femtosecond, 
                                                                          collision_rate=1. / unit.picoseconds,
                                                                          reassign_velocities=True,
                                                                          constraint_tolerance = 1e-6,
                                                                          n_restart_attempts=20),
                                  number_of_iterations=10,
                                  online_analysis_interval=2)

        sampler.setup(n_states=3, mixed_system=system, init_positions=hgv.positions, minimisation_steps=10000, temperature=temperature, storage_file=f"test.nc")  
        sampler.extend(num_iterations)        

        if clean:
            import os
            os.remove("./test.nc")
            os.remove("./test_checkpoint.nc")
            oos.remove("./animodel.pt")
