# This script generates input files for SIRE, needed for EMLE reference energies

import openmm.app
import parmed

pdb = openmm.app.PDBFile("alanine-dipeptide-explicit.pdb")

mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
mm_system = mm_force_field.createSystem(
    pdb.topology, nonbondedMethod=openmm.app.PME, constraints=None, rigidWater=False
)

struct = parmed.openmm.load_topology(pdb.topology, mm_system, xyz=pdb.positions)
# Remove CMAP correction terms, which Sire does not support.
struct.cmaps = parmed.structure.TrackedList()
struct.cmap_types = parmed.structure.TrackedList()
struct.save("alanine-dipeptide-explicit.prmtop", overwrite=True)
struct.save("alanine-dipeptide-explicit.inpcrd", overwrite=True)
