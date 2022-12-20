from copy import deepcopy
import numpy as np
from scipy import sparse
from ase import atoms, io, Atoms, neighborlist
from ase.build import molecule
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

def distances_ab(atoms_a, atoms_b):
    x = atoms_a.get_positions()[:,0] if hasattr(atoms_a, '__len__') else atoms_a.position[0]
    y = atoms_a.get_positions()[:,1] if hasattr(atoms_a, '__len__') else atoms_a.position[1]
    z = atoms_a.get_positions()[:,2] if hasattr(atoms_a, '__len__') else atoms_a.position[2]
    
    u = atoms_b.get_positions()[:,0] if hasattr(atoms_b, '__len__') else atoms_b.position[0]
    v = atoms_b.get_positions()[:,1] if hasattr(atoms_b, '__len__') else atoms_b.position[1]
    w = atoms_b.get_positions()[:,2] if hasattr(atoms_b, '__len__') else atoms_b.position[2]
    
    if hasattr(atoms_a, '__len__') and hasattr(atoms_b, '__len__'):
        if np.shape(atoms_a.get_positions()) != np.shape(atoms_b.get_positions()):
            list_distances = np.zeros((len(x),len(u)))
            for i in range(len(x)):
                list_distances[i,:] = np.sqrt(np.square(x[i]-u) + np.square(y[i]-v) + np.square(z[i]-w))
        else:
            list_distances = np.sqrt(np.square(x-u) + np.square(y-v) + np.square(z-w))
    else:
        list_distances = np.sqrt(np.square(x-u) + np.square(y-v) + np.square(z-w))
    return list_distances

#atoms = io.read('grain_200_gfn2_cycle-200_spherical_hollow.xyz')
atoms = io.read('1000.xyz')

n_components, component_list = atom_to_molecules(atoms) #associate each atoms to its molecule 
mol_coord = np.zeros((n_components,3)) #this will contain every mol barycentre coordinates in x y z
print(n_components)
#mol_to_study = component_list[279] #zx
mol_to_study = component_list[449]
#mol_to_study = component_list[418] #zy
#mol_to_study = 0

for i in range(n_components):
    #for each molecules we obtain the list of atoms contained in it and then compute the barycentre of the molecule and put in it mol_coord
    mol_id = np.where(component_list==i)[0] #search for the id of the atoms contained in the mol

    #for j in range(len(mol_id)):
    atoms_mol = atoms[mol_id] #put the atoms of the molecule in a ase atoms object
    mol_positions = atoms_mol.get_positions() #retrieve the positions of the atoms
    barycentre_positions = np.sum(mol_positions,axis=0)/len(mol_positions) #compute the barycentre of the molecule 
    mol_coord[i,:] = barycentre_positions #associate the positions i of mol_coord to the barycentre position 
    #print(mol_coord[i,:])

#Zprint(mol_coord)

mol_all_distances = np.zeros((3,len(mol_coord),len(mol_coord))) 
mol_all_distances_spherique = np.zeros((3,len(mol_coord),len(mol_coord)))
#mol_all_distances_unit_polar_zx = np.zeros((2,len(mol_coord),len(mol_coord)))
#mol_all_distances_unit_polar_zy = np.zeros((2,len(mol_coord),len(mol_coord)))
#mol_all_distances_unit_polar_xy = np.zeros((2,len(mol_coord),len(mol_coord)))
#mol_all_polar_zx = np.zeros((len(mol_coord),len(mol_coord)))
#mol_all_polar_zy = np.zeros((len(mol_coord),len(mol_coord)))
#mol_all_polar_xy = np.zeros((len(mol_coord),len(mol_coord)))

for i in range(len(mol_coord)):
    #For every mol (i) we are going to compute the spherical coordinates of every mol (j) with i mol at the centre of the sphere
    #Then, to make everything easier, we are going to change the spherical coordinates back to cartesian coordinates
    for j in range(len(mol_coord)):

        mol_distances = mol_coord[j,:] - mol_coord[i,:]
        mol_all_distances[:,i,j] = mol_distances
        
        #we compute the spherical coordinates of j when i is the centre
        mol_all_distances_spherique[0,i,j] = np.sqrt(np.sum(np.square(mol_distances))) #R
        mol_all_distances_spherique[1,i,j] = np.arctan2(np.sqrt(mol_distances[0]**2 + mol_distances[1]**2),mol_distances[2]) #theta
        mol_all_distances_spherique[2,i,j] = np.arctan2(mol_distances[1],mol_distances[0]) #phi

    if len(mol_coord) > 200:
        distance_array_sort = np.zeros((2,len(mol_coord)))
        distance_array_sort[0,:] = np.arange(0,len(mol_coord),1)
        distance_array_sort[1,:] = mol_all_distances_spherique[0,i,:]
        distance_array_sort = distance_array_sort[:,np.argsort(distance_array_sort,axis=1)[1]]
        list_mol = np.sort(distance_array_sort[0,:201].astype(int), axis=0)
    else:
        list_mol = np.arange(0,200,1)

    #print(list_mol)
    print(i)
    if 'mol_all_distances_unit_sphere' not in globals():
        mol_all_distances_unit_sphere = np.zeros((3,len(mol_coord),len(list_mol)))
        mol_all_distances_unit_polar_zx = np.zeros((2,len(mol_coord),len(list_mol)))
        mol_all_distances_unit_polar_zy = np.zeros((2,len(mol_coord),len(list_mol)))
        mol_all_distances_unit_polar_xy = np.zeros((2,len(mol_coord),len(list_mol)))
        mol_all_polar_zx = np.zeros((len(mol_coord),len(list_mol)))
        mol_all_polar_zy = np.zeros((len(mol_coord),len(list_mol)))
        mol_all_polar_xy = np.zeros((len(mol_coord),len(list_mol)))
    
    for jj in range(len(list_mol)):
        
        j = list_mol[jj]
        ii = np.where(list_mol == i)[0][0]
        
        #k = j - i; I move the j coordinate so that i = j is put at 0. it's easier for the next computations
        k = jj - ii

        #print(jj,j,i,ii, k)

        if i == j:
            #for i == j (meaning k == 0) we set every coordinates to 0 since i is the centre of the sphere 
            mol_all_distances_unit_sphere[0,i,k] = 0
            mol_all_distances_unit_sphere[1,i,k] = 0
            mol_all_distances_unit_sphere[2,i,k] = 0

            mol_all_distances_unit_polar_zx[0,i,k] = 0
            mol_all_distances_unit_polar_zx[1,i,k] = 0

            mol_all_distances_unit_polar_zy[0,i,k] = 0
            mol_all_distances_unit_polar_zy[1,i,k] = 0

            mol_all_distances_unit_polar_xy[0,i,k] = 0
            mol_all_distances_unit_polar_xy[1,i,k] = 0

            mol_all_polar_zx[i,k] = 0
            mol_all_polar_zy[i,k] = 0
            mol_all_polar_xy[i,k] = 0
        else:
            #Change back to cartesian but with i at the centre. The R coordinate of the sphere is set to 1 to have a unit sphere
            mol_all_distances_unit_sphere[0,i,k] = np.cos(mol_all_distances_spherique[2,i,j])*np.sin(mol_all_distances_spherique[1,i,j]) #X
            mol_all_distances_unit_sphere[1,i,k] = np.sin(mol_all_distances_spherique[2,i,j])*np.sin(mol_all_distances_spherique[1,i,j]) #Y
            mol_all_distances_unit_sphere[2,i,k] = np.cos(mol_all_distances_spherique[1,i,j])                                            #Z

            #we now project these points into different plane (zx, zy, xy) to analise our data in 2D
            mol_all_distances_unit_polar_zx[0,i,k] = np.cos(mol_all_distances_spherique[2,i,j])*np.sin(mol_all_distances_spherique[1,i,j])
            mol_all_distances_unit_polar_zx[1,i,k] = np.cos(mol_all_distances_spherique[1,i,j])

            mol_all_distances_unit_polar_zy[0,i,k] = np.cos((np.pi/2) - mol_all_distances_spherique[2,i,j])*np.sin(mol_all_distances_spherique[1,i,j])
            mol_all_distances_unit_polar_zy[1,i,k] = np.cos(mol_all_distances_spherique[1,i,j])


            mol_all_distances_unit_polar_xy[0,i,k] = np.cos((np.pi/2) - mol_all_distances_spherique[1,i,j])*np.cos(mol_all_distances_spherique[2,i,j])
            mol_all_distances_unit_polar_xy[1,i,k] = np.cos((np.pi/2) - mol_all_distances_spherique[1,i,j])*np.sin(mol_all_distances_spherique[2,i,j])

            #we compute the theta value for the 3 plane 
            mol_all_polar_zx[i,k] = np.arctan2(mol_all_distances_unit_polar_zx[1,i,k], mol_all_distances_unit_polar_zx[0,i,k])
            mol_all_polar_zy[i,k] = np.arctan2(mol_all_distances_unit_polar_zy[1,i,k], mol_all_distances_unit_polar_zy[0,i,k])
            mol_all_polar_xy[i,k] = np.arctan2(mol_all_distances_unit_polar_xy[1,i,k], mol_all_distances_unit_polar_xy[0,i,k])
        
    #print(mol_all_distances_unit_polar_zx[1,i,:])

        #if mol_all_distances_spherique[2,i,j] == 'nan': print('prout1')

#print(mol_all_distances)
#print(mol_all_distances_spherique)
#print(mol_all_distances[0,0,:])

#fig, ax = plt.subplots()
#fig = plt.figure()
#ax = plt.axes(projection='3d')

#ax.scatter3D(mol_all_distances[0,mol_to_study,:], mol_all_distances[1,mol_to_study,:], mol_all_distances[2,mol_to_study,:], c=mol_all_distances[2,124,:], cmap='Greens');

#ax.scatter3D(mol_all_distances_unit_sphere[0,mol_to_study,:],mol_all_distances_unit_sphere[1,mol_to_study,:],mol_all_distances_unit_sphere[2,mol_to_study,:])

#numb_bins = int(n_components/4)

#The bins will be the same for every plot to keep consistent
numb_bins = 25
bins = [-np.pi + (i/numb_bins)*2*np.pi for i in range(numb_bins + 1)]
#print(len(bins))
#print(bins)

#For every mol we are going to do several plots to better understand the different sets of date we obtained
for i in range(len(mol_coord)):
    print(i)
    #if i not in [0,28]:
    #    continue

    mol_to_study = i
    atoms_to_study = np.where(component_list==i)[0]

    fig = plt.figure(figsize=(12, 8.2))

    ax1 = fig.add_subplot(231) #Plots the x y z coordinates of the XY plane
    ax2 = fig.add_subplot(232) #Plots the x y z coordinates of the XZ plane
    ax3 = fig.add_subplot(233) #Plots the x y z coordinates of the YZ plane
    ax4 = fig.add_subplot(234) #Plots the hist of the theta coordinate for the XY plane (-pi to pi)
    ax5 = fig.add_subplot(235) #Plots the hist of the theta coordinate for the XZ plane (-pi to pi)
    ax6 = fig.add_subplot(236) #Plots the hist of the theta coordinate for the YZ plane (-pi to pi)

    ax1.set_title('XY')
    ax2.set_title('XZ')
    ax3.set_title('YZ')

    fig.suptitle('Mol: ' + str(mol_to_study) +  ' ; Atoms: ' + str(atoms_to_study + 1))

    ax1.set_xlim(-1,1)
    ax1.set_ylim(-1,1)
    ax2.set_xlim(-1,1)
    ax2.set_ylim(-1,1)
    ax3.set_xlim(-1,1)
    ax3.set_ylim(-1,1)

    ax4.set_xlim(-np.pi,np.pi)
    ax5.set_xlim(-np.pi,np.pi)
    ax6.set_xlim(-np.pi,np.pi)

    ax1.scatter(mol_all_distances_unit_polar_xy[0,mol_to_study,1:],mol_all_distances_unit_polar_xy[1,mol_to_study,1:])
    ax2.scatter(mol_all_distances_unit_polar_zx[0,mol_to_study,1:],mol_all_distances_unit_polar_zx[1,mol_to_study,1:])
    ax3.scatter(mol_all_distances_unit_polar_zy[0,mol_to_study,1:],mol_all_distances_unit_polar_zy[1,mol_to_study,1:])

    ax1.scatter(0,0, color='r', zorder=1)
    ax2.scatter(0,0, color='r', zorder=1)
    ax3.scatter(0,0, color='r', zorder=1)

    n1, bins1, patches1 = ax4.hist(mol_all_polar_xy[mol_to_study,1:], bins, density=True)
    n2, bins2, patches2 = ax5.hist(mol_all_polar_zx[mol_to_study,1:], bins, density=True)
    n3, bins3, patches3 = ax6.hist(mol_all_polar_zy[mol_to_study,1:], bins, density=True)

    ax4.set_ylim(0,np.amax(n1))
    ax5.set_ylim(0,np.amax(n2))
    ax6.set_ylim(0,np.amax(n3))

    mean_xy = np.mean(mol_all_polar_xy[mol_to_study,1:])
    std_xy = np.std(mol_all_polar_xy[mol_to_study,1:])
    mean_zx = np.mean(mol_all_polar_zx[mol_to_study,1:])
    std_zx = np.std(mol_all_polar_zx[mol_to_study,1:])
    mean_zy = np.mean(mol_all_polar_zy[mol_to_study,1:])
    std_zy = np.std(mol_all_polar_zy[mol_to_study,1:])

    nbr_n1 = [1 if i > 0 and i < 0.04 else 0 for i in n1]
    nbr_n2 = [1 if i > 0 and i < 0.04 else 0 for i in n2]
    nbr_n3 = [1 if i > 0 and i < 0.04 else 0 for i in n3]

    nbr_n1_0 = [i for i in range(len(n1)) if n1[i] == 0]
    nbr_n1_0_follow_max = 0
    nbr_n1_0_follow = 0
    for l in range(len(nbr_n1_0)):
        if l == 0:
            nbr_n1_0_follow = 1
        elif l > 0:
            if nbr_n1_0[l] == nbr_n1_0[l-1] + 1:
                nbr_n1_0_follow +=1
            else:
                if nbr_n1_0_follow > nbr_n1_0_follow_max:
                    nbr_n1_0_follow_max = deepcopy(nbr_n1_0_follow)
    if nbr_n1_0_follow > nbr_n1_0_follow_max:
        nbr_n1_0_follow_max = deepcopy(nbr_n1_0_follow)

    nbr_n2_0 = [i for i in range(len(n2)) if n2[i] == 0]
    nbr_n2_0_follow_max = 0
    nbr_n2_0_follow = 0    
    for l in range(len(nbr_n2_0)):
        if l == 0:
            nbr_n2_0_follow = 1
        elif l > 0:
            if nbr_n2_0[l] == nbr_n2_0[l-1] + 1:
                nbr_n2_0_follow +=1
            else:
                if nbr_n2_0_follow > nbr_n2_0_follow_max:
                    nbr_n2_0_follow_max = deepcopy(nbr_n2_0_follow)
    if nbr_n2_0_follow > nbr_n2_0_follow_max:
        nbr_n2_0_follow_max = deepcopy(nbr_n2_0_follow)

    nbr_n3_0 = [i for i in range(len(n3)) if n3[i] == 0]
    nbr_n3_0_follow_max = 0
    nbr_n3_0_follow = 0
    for l in range(len(nbr_n3_0)):
        if l == 0:
            nbr_n3_0_follow = 1
        elif l > 0:
            if nbr_n3_0[l] == nbr_n3_0[l-1] + 1:
                nbr_n3_0_follow +=1
            else:
                if nbr_n3_0_follow > nbr_n3_0_follow_max:
                    nbr_n3_0_follow_max = deepcopy(nbr_n3_0_follow)
    if nbr_n3_0_follow > nbr_n3_0_follow_max:
        nbr_n3_0_follow_max = deepcopy(nbr_n3_0_follow)    

    nbr_n1_1 = [1 if i >= 0.04 else 0 for i in n1]
    nbr_n2_1 = [1 if i >= 0.04 else 0 for i in n2]
    nbr_n3_1 = [1 if i >= 0.04 else 0 for i in n3]

    nbr_n1_sum = sum(nbr_n1)
    nbr_n2_sum = sum(nbr_n2)
    nbr_n3_sum = sum(nbr_n3)

    nbr_n1_sum_0 = len(nbr_n1_0)
    nbr_n2_sum_0 = len(nbr_n2_0)
    nbr_n3_sum_0 = len(nbr_n3_0)

    nbr_n1_sum_1 = sum(nbr_n1_1)
    nbr_n2_sum_1 = sum(nbr_n2_1)
    nbr_n3_sum_1 = sum(nbr_n3_1)

    prop_n1_0 = 1 if nbr_n1_sum_0 > 0 else 0 
    prop_n1_1 = (nbr_n1_sum_0 + nbr_n1_sum)/nbr_n1_sum_1
    prop_n2_0 = 1 if nbr_n2_sum_0 > 0 else 0 
    prop_n2_1 = (nbr_n2_sum_0 + nbr_n2_sum)/nbr_n2_sum_1
    prop_n3_0 = 1 if nbr_n3_sum_0 > 0 else 0 
    prop_n3_1 = (nbr_n3_sum_0 + nbr_n3_sum)/nbr_n3_sum_1

    #Compute the integral of the different hist
    int_1 = 1 - (np.sum(n1*(bins[1] - bins[0]))/(np.amax(n1)*2*np.pi))
    int_2 = 1 - (np.sum(n2*(bins[1] - bins[0]))/(np.amax(n2)*2*np.pi))
    int_3 = 1 - (np.sum(n3*(bins[1] - bins[0]))/(np.amax(n3)*2*np.pi))

    ax4.set_title('Nbr n1_0_follow_max = 0: ' + str(nbr_n1_0_follow_max) + '\nInt 1: ' + str(int_1))
    ax5.set_title('Nbr n2_0_follow_max = 0: ' + str(nbr_n2_0_follow_max) + '\nInt 2: ' + str(int_2))
    ax6.set_title('Nbr n3_0_follow_max = 0: ' + str(nbr_n3_0_follow_max) + '\nInt 3: ' + str(int_3))

    #ax4.set_title('Nbr n1 = 0: ' + str(nbr_n1_sum_0) + '\nNbr n1 > 0 and < 0.04: ' + str(nbr_n1_sum) + '\nNbr n1 > 0.04: ' + str(nbr_n1_sum_1) + '\nProp 0: ' + str(prop_n1_0) + '\n prop 1: ' + str(prop_n1_1) + '\nMean: ' + str(mean_xy) + '\nstd: ' + str(std_xy) + '\nInt 1: ' + str(int_1))
    #ax5.set_title('Nbr n2 = 0: ' + str(nbr_n2_sum_0) + '\nNbr n2 > 0 and < 0.04: ' + str(nbr_n2_sum) + '\nNbr n2 > 0.04: ' + str(nbr_n2_sum_1) + '\nProp 0: ' + str(prop_n2_0) + '\n prop 1: ' + str(prop_n2_1) + '\nMean: ' + str(mean_zx) + '\nstd: ' + str(std_zx) + '\nInt 2: ' + str(int_2))
    #ax6.set_title('Nbr n3 = 0: ' + str(nbr_n3_sum_0) + '\nNbr n3 > 0 and < 0.04: ' + str(nbr_n3_sum) + '\nNbr n3 > 0.04: ' + str(nbr_n3_sum_1) + '\nProp 0: ' + str(prop_n3_0) + '\n prop 1: ' + str(prop_n3_1) + '\nMean: ' + str(mean_zy) + '\nstd: ' + str(std_zy) + '\nInt 3: ' + str(int_3))

    if ((nbr_n1_0_follow_max >= 5 and int_1 >= 0.6) or (nbr_n1_0_follow_max >= 3 and int_1 >= 0.65)) or ((nbr_n2_0_follow_max >= 5 and int_2 >= 0.6) or (nbr_n2_0_follow_max >= 3 and int_2 >= 0.65)) or ((nbr_n3_0_follow_max >= 5 and int_3 >= 0.6) or (nbr_n3_0_follow_max >= 3 and int_3 >= 0.65)):
        if 'list_mol_fix' in globals():
            list_mol_fix = np.append(list_mol_fix, i)
        else:
            list_mol_fix = i

    #print('Mol:',i)
    #print(n1)
    #print('xy:', nbr_n1_sum)
    #print(n2)
    #print('xz:', nbr_n2_sum)
    #print(n3)
    #print('yz:', nbr_n3_sum)



    fig.tight_layout()
    #fig.savefig('studies/external_layer_selection/with_k/' + str(mol_to_study) + '.png')
    fig.savefig('studies/external_layer_selection/1000_200-limit/' + str(mol_to_study) + '.png')
    plt.close()
#plt.show()

print(list_mol_fix)

list_atoms_to_fix = np.vstack([np.where(component_list == list_mol_fix[i])[0] for i in range(len(list_mol_fix))])
list_atoms_to_fix = np.reshape(list_atoms_to_fix, list_atoms_to_fix.size)
print(list_atoms_to_fix)

atoms.symbols[list_atoms_to_fix] = 'C'
#io.write('studies/external_layer_selection/with_k/grain_200_gfn2_cycle-200_spherical_hollow_exterior.xyz', atoms)
io.write('studies/external_layer_selection/1000_200-limit/1000_external.xyz', atoms)