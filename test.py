# %%
import pandas as pd
import numpy as np
from ase import io, Atoms
from ase.build import molecule
atoms = io.read('sphere.xyz') 
atoms2 = molecule('H2O')

#print(atoms.get_positions())
print(atoms2.get_positions())
#print(atoms2.get_atomic_numbers())
#print(atoms2.get_chemical_symbols())
#print(atoms2.get_masses()[0])

#position = [(1,1,1),(1,1,1),(1,1,2)]

#atoms2.set_positions(position)

def barycentre(atoms):
    positions = atoms.get_positions()
    barycentre_position = np.array([[np.sum(positions[:,0])/len(positions[:,0])],[np.sum(positions[:,1])/len(positions[:,0])],[np.sum(positions[:,2])/len(positions[:,0])]])
    new_positions = positions - barycentre_position
    atoms.set_positions(new_positions)
    return atoms

atoms2[0].position = atoms2[0].position

print(atoms2.get_positions())

atoms2 = barycentre(atoms2[0:3])

print(atoms2.get_positions())

final_atom = atoms + atoms2



#numpy_df = np.concatenate(((np.array([final_atom.get_chemical_symbols()])).T, final_atom.get_positions()), axis=1)

#df = pd.DataFrame(numpy_df , columns = ['Symbole','X','Y','Z'])

#print(df)

#print(final_atom.get_positions())

#io.write('final.xyz', final_atom)

#io.write('water.xyz', atoms2)
# %%
