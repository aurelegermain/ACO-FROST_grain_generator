import numpy as np
from scipy import sparse
from ase import io, Atoms, neighborlist
from ase.build import molecule
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-structure", "--structure", help="Structure you want to hollow")

args = parser.parse_args()
structure = args.structure
structure_name = Path(structure).stem

def distances_3d(atoms):
    x = atoms.get_positions()[:,0] if hasattr(atoms, '__len__') else atoms.position[0]
    y = atoms.get_positions()[:,1] if hasattr(atoms, '__len__') else atoms.position[1]
    z = atoms.get_positions()[:,2] if hasattr(atoms, '__len__') else atoms.position[2]
    list_distances = np.sqrt(x**2 + y**2 + z**2)
    return list_distances

def atom_to_molecules(atoms):
    cutOff = neighborlist.natural_cutoffs(atoms, mult=0.95)
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=False)
    neighborList.update(atoms)
    matrix = neighborList.get_connectivity_matrix()
    n_components, component_list = sparse.csgraph.connected_components(matrix)
    return n_components, component_list

atoms = io.read(structure)

n_components, component_list = atom_to_molecules(atoms)

list_distances = distances_3d(atoms)

print(n_components, component_list, list_distances)
#np.where(grain_input[:,4]==j)[0]
list_molecules_to_delete = np.unique([component_list[i] for i in range(len(list_distances)) if list_distances[i] < 6])

print(list_molecules_to_delete)

list_atoms_to_delete = np.vstack([np.where(component_list == list_molecules_to_delete[i])[0] for i in range(len(list_molecules_to_delete))])
list_atoms_to_delete = np.reshape(list_atoms_to_delete, list_atoms_to_delete.size)
print(list_atoms_to_delete)

del atoms[list_atoms_to_delete]

io.write(structure_name + '_hollow.xyz', atoms)