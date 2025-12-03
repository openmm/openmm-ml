import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
import os
from sys import stdout

from huggingface_hub import hf_hub_download
model_file_path = hf_hub_download(
    repo_id="Acellera/AceFF-1.0",
    filename="aceff_v1.0.ckpt"
)

pdb = app.PDBFile("../test/data/toluene/toluene.pdb")
potential = MLPotential("TorchMD-Net", modelPath=model_file_path) 
system = potential.createSystem(pdb.topology, charge=0, cudaGraphs=True)

integrator = mm.LangevinIntegrator(
    300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtosecond
)
platform = mm.Platform.getPlatformByName('CUDA')
simulation = app.Simulation(pdb.topology, system, integrator, platform=platform)
simulation.context.setPositions(pdb.positions)
simulation.reporters.append(app.PDBReporter("output.pdb", 100))
simulation.reporters.append(
    app.StateDataReporter(
        stdout, 100, step=True, potentialEnergy=True, temperature=True, speed=True
    )
)

simulation.minimizeEnergy()
simulation.step(1000)
     
