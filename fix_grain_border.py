import numpy as np
from scipy import sparse
from ase import io, Atoms, neighborlist
from ase.build import molecule
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-structure", "--structure", help="Structure you want to hollow") #python3 fix_grain_border.py -structure name_of_the_structure.xyz
parser.add_argument("-distances", "--distances", nargs='+', help="Minimum distances ad maximum distances from the barycentre to fix", type=int, default=[-1,-1])
parser.add_argument("-distances_p", "--distances_percentile", nargs='+', help="Minimum distances and maximum distances from the barycentre to fix in percentiles", type=int, default=[-1,-1])


args = parser.parse_args()
structure = args.structure
list_min_max_distances = args.distances
list_min_max_distances_p = args.distances_percentile
structure_name = Path(structure).stem #strip the .xyz from the structure's name

def distances_3d(atoms):
    """
    Compute the distances from the barycentre for every atoms of the "atoms" object
    """
    x = atoms.get_positions()[:,0] if hasattr(atoms, '__len__') else atoms.position[0]
    y = atoms.get_positions()[:,1] if hasattr(atoms, '__len__') else atoms.position[1]
    z = atoms.get_positions()[:,2] if hasattr(atoms, '__len__') else atoms.position[2]
    list_distances = np.sqrt(x**2 + y**2 + z**2)
    return list_distances

def atom_to_molecules(atoms):
    """
    Takes the "atoms" object and returns a list associating each atoms with a molecule and the number of molecules.
    """
    cutOff = neighborlist.natural_cutoffs(atoms, mult=0.95)
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=False)
    neighborList.update(atoms)
    matrix = neighborList.get_connectivity_matrix()
    n_components, component_list = sparse.csgraph.connected_components(matrix)
    return n_components, component_list

atoms = io.read(structure) #transform the .xyz structure into an atoms object

n_components, component_list = atom_to_molecules(atoms) #associate each atoms to its molecule 

list_distances = distances_3d(atoms) #compute the distances from the barycentre for each atoms

#for each atoms we search which one are farther from "np.percentile(list_distances,50)" but not farther from "np.percentile(list_distances,100)]" and put the index of its molecule into "list_molecules_to_fix"
if list_min_max_distances != [-1,-1]:
    list_molecules_to_fix = np.unique([component_list[i] for i in range(len(list_distances)) if list_distances[i] > list_min_max_distances[0] and list_distances[i] <= list_min_max_distances[1]])
elif list_min_max_distances_p != [-1,-1]:
    list_molecules_to_fix = np.unique([component_list[i] for i in range(len(list_distances)) if list_distances[i] > np.percentile(list_distances,list_min_max_distances_p[0]) and list_distances[i] <= np.percentile(list_distances,list_min_max_distances_p[1])])
else:
    list_molecules_to_fix = np.unique([component_list[i] for i in range(len(list_distances)) if list_distances[i] > np.percentile(list_distances,50) and list_distances[i] <= np.percentile(list_distances,100)])

print(list_molecules_to_fix)

#we convert the list of molecule into a list of atoms
list_atoms_to_fix = np.vstack([np.where(component_list == list_molecules_to_fix[i])[0] for i in range(len(list_molecules_to_fix))])
list_atoms_to_fix = np.reshape(list_atoms_to_fix, list_atoms_to_fix.size)
print(list_atoms_to_fix)

#this tranform each atom to be fixed into a carbon atom to be able to easily see in moldraw or vmd for example which atom is going to be fixed.  
#the name of the file is the name of the xyz file is "name_of_the_structure_fix.xyz"
atoms.symbols[list_atoms_to_fix] = 'C'
io.write(structure_name + '_fix.xyz', atoms)

#Produces an input file using "list_atoms_to_fix" to constrain these atoms using xtb. To be added as xtb --input name_of_the_input_produced.inp name_of_the_structure.xyz --opt -g-fnx > output_file
file_xtb_unfixed_input = open(structure_name + ".inp","w")
print("$constrain", file=file_xtb_unfixed_input)
print("    atoms: ", end="", file=file_xtb_unfixed_input)
#print(list_fix)
list_fix = np.atleast_1d(list_atoms_to_fix)
for k in range(len(list_fix)):
    if k!=0:
        if k==len(list_fix)-1:
            if last_fix == k - 1:
                print("-" + str(list_fix[k]+1), end="", file=file_xtb_unfixed_input)
                j = j + 1
            else:
                print("," + str(list_fix[k]+1), end="", file=file_xtb_unfixed_input)
        else:
            if list_fix[last_fix] == list_fix[k] - 1:
                last_fix = k
            else:
                print("-" + str(list_fix[last_fix]+1) + "," + str(list_fix[k]+1), end="", file=file_xtb_unfixed_input)
                last_fix = k
                j = j + 1
    elif k==0:
        j = 0
        last_fix = k
        print(list_fix[k]+1, end="", file=file_xtb_unfixed_input)
print("\n$end", file=file_xtb_unfixed_input)
file_xtb_unfixed_input.close()