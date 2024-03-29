# %%
import pandas as pd
from copy import deepcopy
import subprocess
import numpy as np
from scipy import sparse
from ase import io, Atoms, neighborlist, geometry
from ase.build import molecule
from ase.data.pubchem import pubchem_atoms_search, pubchem_atoms_conformer_search
import argparse
import os
import sys
import math
from datetime import datetime
startTime = datetime.now()

rng =np.random.default_rng()

parser = argparse.ArgumentParser()
parser.add_argument("-size", "--size", help="Size", type=int, default="0")
parser.add_argument("-g", "--gfn", help="GFN-xTB method to use", default="2")
parser.add_argument("-structure", "--structure", help="Pre-existing structure")
parser.add_argument("-fixed_structure", "--fixed_structure", help="Pre-existing structure", action='store_true')
parser.add_argument("-mix", "--mixed", nargs='+', help="If you want to produce a grain using the mixed method. High accuracy level first, second is every x molecules the high level is done. Low level is what was chosen with --gfn. For ff as low level we add one high level at 50.", type=str)
parser.add_argument("-inp", "--input", help="Give the name of the input file from which the program will read the parameters of the grain building process.", type=str)
parser.add_argument("-mol", "--molecule", help="If you want a grain with only one type of molecule.", type=str, default="H2O")
parser.add_argument("-r", "--restart", help="If it is a restart and from where it is restarting", type=int)
parser.add_argument("-md", "--MD", nargs='+', help="Molecular dynamics: Which method and every x molecule", type=str)
parser.add_argument("-agermain2022", "--agermain2022", help="Produce a grain using the method described in Germain et al. 2022 to be submitted. Only needs -size.", action='store_true')
parser.add_argument("-rand", "--random_law", help="Which rule to use for the random position of molecules", default="normal")
parser.add_argument("-opt_cycle", "--optimisation_cycle", help="Number of molecules added before optimisation.", type=int, default="1")
parser.add_argument("-final_gfn2", "--final_gfn2", help="A GFN2 optimisation will be done at the end of the building process.", default=False, action='store_true')



args = parser.parse_args()

input_file = args.input

size = args.size
restart = args.restart

gfn = str(args.gfn)
structure = args.structure
fixed_structure = args.fixed_structure
mol = str(args.molecule)
agermain2021 = args.agermain2022
random_law = args.random_law
opt_cycle = args.optimisation_cycle
final_gfn2 = args.final_gfn2

check_surface = False

fixed_core_input = None

if input_file is not None:

    with open(input_file , "rt") as myfile:
        output = myfile.read()

    try:    
        start_string="$parameters"
        end_string="$end" 
        start = output.index(start_string) + len(start_string)
        end = output.index(end_string, start)
        input_parameters = output[start:end].split('\n')

        for i,line in enumerate(input_parameters):

            if 'distrib' in line:
                distrib = int(line.split()[1])
           
            if 'gfn ' in line:
                gfn = str(line.split()[1])
            
            if 'size' in line:
                size = int(line.split()[1])

            if 'mixed' in line:
                High_method_and_cycle = str(line.split()[1:3])

            if 'MD' in line:
                MD_method_and_cycle = str(line.split()[1:3])
            
            if 'struct ' in line:
                structure = str(line.split()[1])

            if 'fixstruct' in line:
                fixed_structure = str(line.split()[1])

            if 'restart' in line:
                restart = int(line.split()[1])

            if 'rand' in line:
                random_law = str(line.split()[1])

            if 'opt' in line:
                opt_cycle = int(line.split()[1])

            if 'check' in line:
                check_surface = True

            if 'finalgfn2' in line:
                final_gfn2 = True

            if 'agermain2022' in line:
                agermain2021 = True

    except:
        input_parameters = None

if final_gfn2 == True:
    Time_file = open('grain_' + str(size) + '_gfn' + str(gfn) + '_cycle-' + str(opt_cycle) + '_gfn2_execution_time.txt', 'w')
else:
    Time_file = open('grain_' + str(size) + '_gfn' + str(gfn) + '_cycle-' + str(opt_cycle) + '_execution_time.txt', 'w')

MD_method_and_cycle = args.MD
if MD_method_and_cycle is not None:
    MD_method = str(MD_method_and_cycle[0])
    MD_cycle = int(MD_method_and_cycle[1])

High_method_and_cycle = args.mixed
if High_method_and_cycle is not None:
    High_method = str(High_method_and_cycle[0])
    High_cycle = int(High_method_and_cycle[1])
    if structure is not None:
        atoms = io.read(structure)
        if fixed_structure is True:
            file_xtb_input = open("./MD.inp","w")
            print("$fix", file=file_xtb_input)
            print("    atoms: 1-" + str(len(atoms)), file=file_xtb_input)
            print("$end", file=file_xtb_input)
            print('$md\n   temp=10 # in K\n   time=1\n   dump=25\n   step=  1.0  # in fs\n$end',file=file_xtb_input)
            file_xtb_input.close()
    else:
        file_md_inp = open('MD.inp', 'w')
        print('$md\n   temp=10 # in K\n   time=1\n   dump=25\n   step=  1.0  # in fs\n$end',file=file_md_inp)
        file_md_inp.close()

distance = 2.5
coeff_min = 1.00
coeff_max = 1.1
steps = 0.1
nbr_not_converged = 0

if agermain2021 is not False:
    check_surface = True
    final_gfn2 = True
    MD_method_and_cycle = 0
    MD_method = 'ff'
    MD_cycle = 10
    High_method_and_cycle = 0
    High_method = '2'
    High_cycle = 100
    gfn = 'ff'
    mol = 'H2O'
    if structure is not None:
        atoms = io.read(structure)
        if fixed_structure is True:
            file_xtb_input = open("./MD.inp","w")
            print("$fix", file=file_xtb_input)
            print("    atoms: 1-" + str(len(atoms)), file=file_xtb_input)
            print("$end", file=file_xtb_input)
            print('$md\n   temp=10 # in K\n   time=1\n   dump=25\n   step=  1.0  # in fs\n$end',file=file_xtb_input)
            file_xtb_input.close()
    else:
        file_md_inp = open('MD.inp', 'w')
        print('$md\n   temp=10 # in K\n   time=1\n   dump=25\n   step=  1.0  # in fs\n$end',file=file_md_inp)
        file_md_inp.close()


if check_surface == True:
    nbr_final_gfn2 = 0
    nbr_single_hb = 0
    nbr_mol_attrib_problem = 0

if input_file is not None:

    with open(input_file , "rt") as myfile:
        output = myfile.read()

    try:    
        start_string="$building"
        end_string="$end" 
        start = output.index(start_string) + len(start_string)
        end = output.index(end_string, start)
        input_building = output[start:end].strip().split()
        input_building = np.reshape(input_building,(-1,2))
    except:
        input_building = None

    try:    
        start_string="$orca"
        end_string="$end" 
        start = output.index(start_string) + len(start_string)
        end = output.index(end_string, start)
        input_orca = output[start:end].strip().split()
        input_orca = np.reshape(input_orca,(-1,2))
    except:
        input_orca = None
    
    if input_building is not None:
        list_mol = deepcopy(input_building)
        size_partial = np.sum(list_mol[:,1].astype(int))

        if size_partial == 100:
            if 'size' in globals() and size > 100:
                list_mol[:,1] = (list_mol[:,1]*(size/100)).astype(int)
        else:
            size = size_partial

        if 'distrib' in globals():
            if distrib == 1:
                method_selection = 'random'
            else:
                method_selection = 'follow'
        else:
            distrib = 0
            method_selection = 'follow'
else:
    list_mol = np.array([mol, size])

def FromXYZtoDataframeMolecule(input_file):
    #read and encode .xyz fiel into Pandas Dataframe
    df_xyz = pd.DataFrame(columns = ['Atom','X','Y','Z'])
    with open(input_file, "r") as data:
        lines = data.readlines()
        for i in range(2,len(lines)):
            line = lines[i].split()
            if len(line) == 4:
                new_row = pd.Series({'Atom':line[0],'X':line[1],'Y':line[2],'Z':line[3]},name=3)
                df_xyz = df_xyz.append(new_row,ignore_index=True)
    df_xyz['Atom'].astype(str)
    df_xyz['X'].astype(float)
    df_xyz['Y'].astype(float)
    df_xyz['Z'].astype(float)
    #compute neighbor of the atoms in xyz format with ASE package
    mol = io.read(input_file)
    cutOff = neighborlist.natural_cutoffs(mol)
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    neighborList.update(mol)
    #create molecules column in dataframe related to connectivity, molecules start from 0
    mol_vector = np.zeros(df_xyz.shape[0],dtype=int)
    for i in range(df_xyz.shape[0]):
        if i == 0:
            mol_ix = 1
            n_list = neighborList.get_neighbors(i)[0]
            mol_vector[i] = mol_ix
            for j,item_j in enumerate(n_list):
                mol_vector[item_j] = mol_ix
                j_list = neighborList.get_neighbors(item_j)[0]
                j_list = list(set(j_list) - set(n_list))
                for k,item_k in enumerate(j_list):
                    mol_vector[item_k] = mol_ix
        elif mol_vector[i] == 0:
            mol_ix = mol_ix + 1 
            n_list = neighborList.get_neighbors(i)[0]
            mol_vector[i] = mol_ix
            for j,item_j in enumerate(n_list):
                mol_vector[item_j] = mol_ix
                j_list = neighborList.get_neighbors(item_j)[0]
                j_list = list(set(j_list) - set(n_list))
                for k,item_k in enumerate(j_list):
                    mol_vector[item_k] = mol_ix
    mol_vector = mol_vector - 1
    df_xyz['Molecules'] = mol_vector
    return(df_xyz)

def LabelMoleculesRadius(df_xyz,mol_ref,radius):
    df_xyz['X'].astype(float)
    df_xyz['Y'].astype(float)
    df_xyz['Z'].astype(float)
    n_mol = df_xyz['Molecules'].max() + 1
    mol_vector = np.array(list(range(n_mol)),dtype=int)
    X_c = np.zeros(n_mol,dtype=float)
    Y_c = np.zeros(n_mol,dtype=float)
    Z_c = np.zeros(n_mol,dtype=float)
    for i in range(n_mol):
        df_tmp = df_xyz[df_xyz['Molecules'] == i]
        X_c[i] = df_tmp['X'].astype(float).mean()
        Y_c[i] = df_tmp['Y'].astype(float).mean()
        Z_c[i] = df_tmp['Z'].astype(float).mean()
    df_center = pd.DataFrame()
    df_center['Molecules'] = mol_vector
    df_center['X'] = X_c
    df_center['Y'] = Y_c
    df_center['Z'] = Z_c
    tmp_c = df_center[df_center['Molecules'] == mol_ref].values[0]
    dist_vector = np.zeros(n_mol,dtype=float)
    dist_bool = np.full(n_mol, False)
    for index, rows in df_center.iterrows():
        dist_vector[index] = math.sqrt((rows.X - tmp_c[1])**2 + (rows.Y - tmp_c[2])**2 + (rows.Z - tmp_c[3])**2)
        if dist_vector[index] < radius:
            dist_bool[index] = True
    df_center['Distance'] = dist_vector
    df_center['Shell'] = dist_bool
    xyz_bool = np.full(df_xyz.shape[0], 'M')
    for index, rows in df_xyz.iterrows():
        if df_center[df_center['Molecules'] == rows.Molecules].values[0][5] == True:
            xyz_bool[index] = 'H'    
    df_xyz['Level'] = xyz_bool
    return(df_xyz)

def FromXYZtoMoleculeNumpy(input_file):
    NumpyMolecule = FromXYZtoDataframeMolecule(input_file).to_numpy() 
    return NumpyMolecule

def atom_to_molecules(atoms):
    cutOff = neighborlist.natural_cutoffs(atoms, mult=0.95)
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=False)
    neighborList.update(atoms)
    matrix = neighborList.get_connectivity_matrix()
    n_components, component_list = sparse.csgraph.connected_components(matrix)
    return n_components, component_list

def radius_gyration(atoms):
    radius_gyration = np.sqrt(np.dot(atoms.get_masses(), distances_3d(atoms.get_positions())**2) / np.sum(atoms.get_masses()))
    return radius_gyration

def distances_3d(atoms):
    x = atoms.get_positions()[:,0] if hasattr(atoms, '__len__') else atoms.position[0]
    y = atoms.get_positions()[:,1] if hasattr(atoms, '__len__') else atoms.position[1]
    z = atoms.get_positions()[:,2] if hasattr(atoms, '__len__') else atoms.position[2]
    list_distances = np.sqrt(x**2 + y**2 + z**2)
    return list_distances

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

def barycentre(atoms):
    positions = atoms.get_positions()
    barycentre_position = np.array([np.sum(positions[:,0])/len(positions[:,0]),np.sum(positions[:,1])/len(positions[:,0]),np.sum(positions[:,2])/len(positions[:,0])])
    new_positions = np.zeros(np.shape(positions))
    for i in range(len(new_positions)):
        new_positions[i,:] = positions[i,:] - barycentre_position
    atoms.set_positions(new_positions)
    return atoms

def molecule_closer(sphere, Atoms_to_add, distance_grain):
    good_place = False
    start_again = 0
    while good_place!=True:
        All_distances_from_grain = geometry.get_distances(Atoms_to_add.get_positions(), sphere.get_positions())[1]
        id_mol, id_nearest = np.divmod(np.argmin(All_distances_from_grain),len(sphere))
        mol_position = Atoms_to_add[id_mol].position
        nearest_position = sphere[id_nearest].position

        delta_a = np.sum(mol_position**2)
        delta_b = -2*np.sum(mol_position*nearest_position)
        delta_c = np.sum(nearest_position**2) - distance_grain**2
        delta = delta_b**2 -4*delta_a*delta_c

        if delta > 0:
            t_0 = (-delta_b + np.sqrt(delta))/(2*delta_a)
            t_1 = (-delta_b - np.sqrt(delta))/(2*delta_a)
            mol_position_0 = np.array([t_0*mol_position[0], t_0*mol_position[1], t_0*mol_position[2]])
            mol_position_1 = np.array([t_1*mol_position[0], t_1*mol_position[1], t_1*mol_position[2]])
            d_mol_0 = np.sqrt(np.sum((mol_position_0)**2))
            d_mol_1 = np.sqrt(np.sum((mol_position_1)**2))

            if d_mol_0 > d_mol_1: 
                diff_mol_position = mol_position_0 - mol_position
            else:
                diff_mol_position = mol_position_1 - mol_position
        else:
            delta_c_2 = [np.sum(nearest_position**2) - (i/10)**2 for i in range(20,40,1)]
            delta_2 = [delta_b**2 -4*delta_a*delta_c_i for delta_c_i in delta_c_2]

            if start_again > 5:
                t_2_0 = (-delta_b + np.sqrt(np.abs(delta)))/(2*delta_a)
                t_2_1 = (-delta_b - np.sqrt(np.abs(delta)))/(2*delta_a)
            
                mol_position_2_0 = np.array([t_2_0*mol_position[0], t_2_0*mol_position[1], t_2_0*mol_position[2]])
                mol_position_2_1 = np.array([t_2_1*mol_position[0], t_2_1*mol_position[1], t_2_1*mol_position[2]])
                d_mol_2_0 = np.sqrt(np.sum((mol_position_2_0)**2))
                d_mol_2_1 = np.sqrt(np.sum((mol_position_2_1)**2))
                if d_mol_2_0 < d_mol_2_1: 
                    diff_mol_position = mol_position_2_0 - mol_position
                else:
                    diff_mol_position = mol_position_2_1 - mol_position
            else:
                t_2 = (-delta_b)/(2*delta_a)
                mol_position_2 = np.array([t_2*mol_position[0], t_2*mol_position[1], t_2*mol_position[2]])
                diff_mol_position = mol_position_2 - mol_position
        Atoms_to_add_0 = deepcopy(Atoms_to_add)
        Atoms_to_add.set_positions(Atoms_to_add.get_positions() + diff_mol_position)
        list_min_distances_from_grain = [np.argmin(distances) for distances in All_distances_from_grain] + [np.argmin(All_distances_from_grain)]
        All_distances_from_grain_1 = geometry.get_distances(Atoms_to_add.get_positions(), sphere.get_positions())[1]
        
        if ((np.amin(All_distances_from_grain_1) > distance_grain*0.95) and (np.amin(All_distances_from_grain_1) < distance_grain*1.05)) or start_again > 10:
            good_place = True
            #print(i, start_again, np.amin(All_distances_from_grain_1), delta)
            return Atoms_to_add
        else:
            #print(i, start_again, np.amin(All_distances_from_grain_1), delta)
            start_again += 1

def molecule_positioning_simplified(atoms, name_atom_added, random_law):
    atoms2 = deepcopy(io.read('./' + name_atom_added + '_gfn' + gfn + '/' + name_atom_added + '.xyz'))
    barycentre(atoms2)

    if random_law == "normal":
        angle_mol = rng.random(3)*360
        atoms2.rotate(angle_mol[0], "x")
        atoms2.rotate(angle_mol[1], "y")
        atoms2.rotate(angle_mol[2], "z")
        positions = atoms2.get_positions()

        theta = rng.random()*2 - 1
        phi = rng.random()*2*np.pi

        radius = np.percentile(distances_3d(atoms),90)

        position_mol = np.zeros(3)

        position_mol[0] = (radius + distance)*np.sqrt(1 - theta**2)*np.cos(phi)
        position_mol[1] = (radius + distance)*np.sqrt(1 - theta**2)*np.sin(phi)
        position_mol[2] = (radius + distance)*theta

        atoms2.set_positions(positions + position_mol)

        atoms2 = molecule_closer(atoms, atoms2, distance)

    elif random_law == "sbiased":
        angle_mol = rng.random(3)*360
        atoms2.rotate(angle_mol[0], "x")
        atoms2.rotate(angle_mol[1], "y")
        atoms2.rotate(angle_mol[2], "z")
        positions = atoms2.get_positions()

        theta = rng.random()*np.pi
        phi = rng.random()*2*np.pi

        radius = np.percentile(distances_3d(atoms),90)

        position_mol = np.zeros(3)

        position_mol[0] = (radius + distance)*np.cos(phi)*np.sin(theta)
        position_mol[1] = (radius + distance)*np.sin(phi)*np.sin(theta)
        position_mol[2] = (radius + distance)*np.cos(theta)

        atoms2.set_positions(positions + position_mol)

        atoms2 = molecule_closer(atoms, atoms2, distance)

    #elif random_law == "grid":
    elif random_law == "plane":
        angle_mol = rng.random(3)*360
        atoms2.rotate(angle_mol[0], "x")
        atoms2.rotate(angle_mol[1], "y")
        atoms2.rotate(angle_mol[2], "z")
        positions = atoms2.get_positions()

        theta = rng.random()*2*np.pi
        phi = 0

        radius = np.percentile(distances_3d(atoms),90)

        position_mol = np.zeros(3)

        position_mol[0] = (radius + distance)*np.cos(phi)*np.sin(theta)
        position_mol[1] = (radius + distance)*np.sin(phi)*np.sin(theta)
        position_mol[2] = (radius + distance)*np.cos(theta)

        atoms2.set_positions(positions + position_mol)
        atoms2 = molecule_closer(atoms, atoms2, distance)

    elif random_law == "hollow":
        #distance = 2
        hollow_radius = 3
        if len(atoms) == len(atoms2): atoms.set_positions(atoms.get_positions() + hollow_radius) #A changer!!!
        angle_mol = rng.random(3)*360
        atoms2.rotate(angle_mol[0], "x")
        atoms2.rotate(angle_mol[1], "y")
        atoms2.rotate(angle_mol[2], "z")
        positions = atoms2.get_positions()

        theta = rng.random()*2 - 1
        phi = rng.random()*2*np.pi

        radius = np.percentile(distances_3d(atoms),50)

        position_mol = np.zeros(3)

        position_mol[0] = (hollow_radius)*np.sqrt(1 - theta**2)*np.cos(phi)
        position_mol[1] = (hollow_radius)*np.sqrt(1 - theta**2)*np.sin(phi)
        position_mol[2] = (hollow_radius)*theta

        atoms2.set_positions(positions + position_mol)

        i = 0
        while np.amin(distances_ab(atoms2, atoms)) < distance / coeff_min or np.amin(distances_3d(atoms2)) < hollow_radius:
            #print(i)
            i = i + 1
            
            position_mol[0] = (hollow_radius + i*steps)*np.sqrt(1 - theta**2)*np.cos(phi)
            position_mol[1] = (hollow_radius + i*steps)*np.sqrt(1 - theta**2)*np.sin(phi)
            position_mol[2] = (hollow_radius + i*steps)*theta

            atoms2.set_positions(positions + position_mol)
    elif random_law == "spherical":
        theta = 2
        phi = 2.5
        phi_level_list = 1
        index = 0
        k=0
        atoms_size = atom_to_molecules(atoms)[0]
        atoms_to_add = deepcopy(atoms2)
        atoms_to_add2 = 0

        while index < size:
            if index == (atoms_size + 1) and index < size: 
                break

            if k != 0:
                theta = int(k*2 + 1)
                phi = int(k*4)
                phi_level_list = np.append(phi_level_list,phi)
                position_mol = np.zeros(3)
                theta_list = [(i/(theta-1))*np.pi for i in range(theta)]
                radius = k*3  
                for i in range(len(theta_list)):
                    if index == (atoms_size + 1) and index < size: 
                        break
                    phi_level = int(abs(abs(i - (len(theta_list)-1)/2) - (len(theta_list)-1)/2))
                    phi_list = [(l/phi_level_list[phi_level])*2*np.pi for l in range(phi_level_list[phi_level])]

                    for j in range(len(phi_list)):
                        atoms_to_add = deepcopy(atoms2)
                        position_mol[0] = (radius)*np.cos(phi_list[j])*np.sin(theta_list[i])
                        position_mol[1] = (radius)*np.sin(phi_list[j])*np.sin(theta_list[i])
                        position_mol[2] = (radius)*np.cos(theta_list[i])

                        index +=1

                        if index == (atoms_size + 1) and index < size:
                            #if atoms_to_add2 !=0: 
                            #    print('i:', i, 'j:', j, 'len(theta_list):', len(theta_list), 'len(phi_list):', len(phi_list), 'index:', index, 'atoms_size:', atoms_size, 'atoms_to_add2 size', atom_to_molecules(atoms_to_add2)[0], 'break')
                            #else:
                            #    print('i:', i, 'j:', j, 'len(theta_list):', len(theta_list), 'len(phi_list):', len(phi_list), 'index:', index, 'atoms_size:', atoms_size, 'atoms_to_add2 size', '0', 'break')
                            break
                        elif index >= (atoms_size + 1) and index >=size:
                            if (j < (len(phi_list) ) and i < (len(theta_list))) or (j < (len(phi_list)) and i == (len(theta_list) - 1)):
                                positions = atoms_to_add.get_positions() 
                                atoms_to_add.set_positions(positions + position_mol)
                                #if atoms_to_add2 != 0:
                                #    print('i:', i, 'j:', j, 'len(theta_list):', len(theta_list), 'len(phi_list):', len(phi_list), 'index:', index, 'atoms_size:', atoms_size, 'atoms_to_add2 size', atom_to_molecules(atoms_to_add2)[0])
                                #else:
                                #    print('i:', i, 'j:', j, 'len(theta_list):', len(theta_list), 'len(phi_list):', len(phi_list), 'index:', index, 'atoms_size:', atoms_size, 'atoms_to_add2 size', '0')
                                if atoms_to_add2 == 0:
                                    atoms_to_add2 = deepcopy(atoms_to_add)
                                else:
                                    atoms_to_add2 = atoms_to_add2 + atoms_to_add
                                    #print('A atoms_to_add2 size:', atom_to_molecules(atoms_to_add2)[0])
                k += 1
            elif k == 0:
                k += 1
                index += 1   
        positions = atoms_to_add.get_positions() 
        atoms_to_add.set_positions(positions + position_mol)
        if atoms_to_add2 == 0:
            atoms_to_add2 = deepcopy(atoms_to_add)
        atoms2 = deepcopy(atoms_to_add2)
        #print('B atoms_to_add2 size:', atom_to_molecules(atoms_to_add2)[0])

    return atoms2

def check_surface_agermain2021(atoms, i, nbr_mol, nbr_final_gfn2, nbr_mol_attrib_problem, nbr_single_hb):

    def magnitude_vectors(x, y, z, u, v, w):
        x = x.astype(float) if hasattr(x, '__len__') and len(x) > 1 else float(x)
        y = y.astype(float) if hasattr(y, '__len__') and len(y) > 1 else float(y)
        z = z.astype(float) if hasattr(z, '__len__') and len(z) > 1 else float(z)

        u = u.astype(float) if hasattr(u, '__len__') and len(u) > 1 else float(u)
        v = v.astype(float) if hasattr(v, '__len__') and len(v) > 1 else float(v)
        w = w.astype(float) if hasattr(w, '__len__') and len(w) > 1 else float(w)

        magnitude = np.sqrt(np.square(x-u) + np.square(y-v) + np.square(z-w))
        return magnitude

    D_min_H_bond = 1.4
    D_max_H_bond = 2.1
    coeff_two_hbonds = 1.5
    nbr_weird_O = 0
    restart_mol_attrib_problem = False

    input_grain_check_file  = './check_surface.xyz'
    io.write(input_grain_check_file, atoms)

    grain_input = FromXYZtoMoleculeNumpy(input_grain_check_file)
    
    nbr_mol_file = np.amax(grain_input[:,4]) + 1
    nbr_atom = nbr_mol_file*3
    
    #print(nbr_mol_file)
    #print(grain_input)
    
    list_nbr_h_bond = np.zeros(nbr_mol_file)
    
    for j in range(nbr_mol_file):
        oxygen_1 = grain_input[(grain_input[:,0]=='O') & (grain_input[:,4]==j)]
        hydrogen_1 = grain_input[(grain_input[:,0]=='H') & (grain_input[:,4]!=j)]
        oxygen_2 = grain_input[(grain_input[:,0]=='O') & (grain_input[:,4]!=j)]
        hydrogen_2 = grain_input[(grain_input[:,0]=='H') & (grain_input[:,4]==j)]
        if oxygen_1.shape[0]!=1 or hydrogen_2.shape[0]!=2:
            nbr_weird_O = nbr_weird_O + int(oxygen_1.shape[0])
            if "list_weird_mol" in locals():
                list_weird_mol = np.append(list_weird_mol, j)
            else:
                list_weird_mol = np.array([j])
                list_weird_mol = list_weird_mol.tolist()
    if "list_weird_mol" in locals():
        print("Weird Mol:" + str(list_weird_mol))
         
        os.system("mkdir final_gfn2_" + str(nbr_final_gfn2))
        
        for j in range(nbr_mol - nbr_weird_O + 1,nbr_mol+1):
            os.system("mv " + str(j) + "* ./final_gfn2_" + str(nbr_final_gfn2))
        os.system("mv " + input_grain_check_file + " ./final_gfn2_" + str(nbr_final_gfn2))
            
        for j in list_weird_mol:
            grain_input = np.delete(grain_input, np.where(grain_input[:,4]==j)[0], axis=0) 
            
        new_file = open("sans_weird_mol_grain.xyz","w")
        n_mol_new_file = grain_input[:,0]
        x_mol_new_file = grain_input[:,1].astype(float)
        y_mol_new_file = grain_input[:,2].astype(float)
        z_mol_new_file = grain_input[:,3].astype(float)
        for j in range(len(n_mol_new_file)):
            if j==0:
                print(str(len(n_mol_new_file)) + "\n", file=new_file)
            n_mol_new_file_j = n_mol_new_file[j]
            x_mol_new_file_j = x_mol_new_file[j]
            y_mol_new_file_j = y_mol_new_file[j]
            z_mol_new_file_j = z_mol_new_file[j]
                 
            print(n_mol_new_file_j + str("{: {width}.{prec}f}".format(x_mol_new_file_j, width=25, prec=14)) + str("{: {width}.{prec}f}".format(y_mol_new_file_j, width=20, prec=14)) + str("{: {width}.{prec}f}".format(z_mol_new_file_j, width=20, prec=14)), file=new_file)
        new_file.close()
        
        i = i - nbr_weird_O
        nbr_final_gfn2 = nbr_final_gfn2 + 1
        nbr_weird_O = 0
        del list_weird_mol
        atoms_return = io.read("sans_weird_mol_grain.xyz")
        os.system("rm sans_weird_mol_grain.xyz")
        return atoms_return, i, nbr_final_gfn2, nbr_mol_attrib_problem, nbr_single_hb
        
    
    for j in range(nbr_mol_file):
        oxygen_1 = grain_input[(grain_input[:,0]=='O') & (grain_input[:,4]==j)]
        hydrogen_1 = grain_input[(grain_input[:,0]=='H') & (grain_input[:,4]!=j)]
        oxygen_2 = grain_input[(grain_input[:,0]=='O') & (grain_input[:,4]!=j)]
        hydrogen_2 = grain_input[(grain_input[:,0]=='H') & (grain_input[:,4]==j)]
        if oxygen_1.shape[0]!=1 or hydrogen_2.shape[0]!=2:
            print("probleme")
            print(oxygen_1.shape[0], hydrogen_2.shape[0])
            restart_mol_attrib_problem = True
            break
        d_oxygenhydrogen_1 = magnitude_vectors(oxygen_1[:,1].astype(float), oxygen_1[:,2].astype(float), oxygen_1[:,3].astype(float), hydrogen_1[:,1].astype(float), hydrogen_1[:,2].astype(float), hydrogen_1[:,3].astype(float))
        d_oxygenhydrogen_2 = magnitude_vectors(oxygen_2[:,1].astype(float), oxygen_2[:,2].astype(float), oxygen_2[:,3].astype(float), np.float(hydrogen_2[0,1]), np.float(hydrogen_2[0,2]), np.float(hydrogen_2[0,3]))
        d_oxygenhydrogen_3 = magnitude_vectors(oxygen_2[:,1].astype(float), oxygen_2[:,2].astype(float), oxygen_2[:,3].astype(float), np.float(hydrogen_2[1,1]), np.float(hydrogen_2[1,2]), np.float(hydrogen_2[1,3]))
        h_bond_1 = d_oxygenhydrogen_1[d_oxygenhydrogen_1 <= D_max_H_bond]
        h_bond_2 = d_oxygenhydrogen_2[d_oxygenhydrogen_2 <= D_max_H_bond]
        h_bond_3 = d_oxygenhydrogen_3[d_oxygenhydrogen_3 <= D_max_H_bond]
        list_nbr_h_bond[j] = list_nbr_h_bond[j] + h_bond_1.size + h_bond_2.size + h_bond_3.size
    if restart_mol_attrib_problem==True:
        print("problem")
        print(i, nbr_mol)
        os.system("mkdir " + str(i) + "-mol-attrib-problem-" + str(nbr_mol_attrib_problem))
        os.system("mv " + str(i) + "* ./" + str(i) + "-mol-attrib-problem-" + str(nbr_mol_attrib_problem))
        atoms_return = io.read("./" + str(i - 1) + "/xtbopt.xyz")
        nbr_mol_attrib_problem = nbr_mol_attrib_problem + 1
        restart_mol_attrib_problem = False
        return atoms_return, i-1, nbr_final_gfn2, nbr_mol_attrib_problem, nbr_single_hb
    
    print(list_nbr_h_bond)   
    
    m, = np.where(list_nbr_h_bond<=1)
    
    print(m)
    
    n, = np.where(np.isin(grain_input[:,4], m))
    print(n)
    
    two_hbonds, = np.where(list_nbr_h_bond==2)
    
    print(two_hbonds)
    
    list_two_hbonds = np.zeros([two_hbonds.size, 2]).astype(int)
    
    for k in range(len(two_hbonds)):
    
        oxygen = grain_input[(grain_input[:,0]=='O') & (grain_input[:,4]==two_hbonds[k])]
        hydrogen = grain_input[(grain_input[:,0]=='H') & (grain_input[:,4]!=two_hbonds[k])]
        d_oxygenhydrogen = magnitude_vectors(oxygen[:,1].astype(float), oxygen[:,2].astype(float), oxygen[:,3].astype(float), hydrogen[:,1].astype(float), hydrogen[:,2].astype(float), hydrogen[:,3].astype(float))
    
        h_bond, = hydrogen[np.where(d_oxygenhydrogen <= D_max_H_bond),4]
    
        if h_bond.size!=0:
            if h_bond.size==1:
                place = 0
                list_two_hbonds[k,place] = h_bond
                place = 1
            elif h_bond.size==2:
                list_two_hbonds[k,0] = h_bond[0]
                list_two_hbonds[k,1] = h_bond[1]
                continue
            
        oxygen = grain_input[(grain_input[:,0]=='O') & (grain_input[:,4]!=two_hbonds[k])]
        hydrogen = grain_input[(grain_input[:,0]=='H') & (grain_input[:,4]==two_hbonds[k])]
            
        d_oxygenhydrogen = magnitude_vectors(oxygen[:,1].astype(float), oxygen[:,2].astype(float), oxygen[:,3].astype(float), np.float(hydrogen[0,1]), np.float(hydrogen[0,2]), np.float(hydrogen[0,3]))
    
        h_bond, = oxygen[np.where(d_oxygenhydrogen <= D_max_H_bond),4]
    
        if h_bond.size!=0:
            if h_bond.size==1:
                if "place" in locals():
                    list_two_hbonds[k,place] = h_bond
                    del place
                    continue
                else:
                    place = 0
                    list_two_hbonds[k,place] = h_bond
                    place = 1
            elif h_bond.size==2:
                list_two_hbonds[k,0] = h_bond[0]
                list_two_hbonds[k,1] = h_bond[1]
                continue
            
        d_oxygenhydrogen = magnitude_vectors(oxygen[:,1].astype(float), oxygen[:,2].astype(float), oxygen[:,3].astype(float), np.float(hydrogen[1,1]), np.float(hydrogen[1,2]), np.float(hydrogen[1,3]))
    
        h_bond, = oxygen[np.where(d_oxygenhydrogen <= D_max_H_bond),4]
    
        if h_bond.size!=0:
            if h_bond.size==1:
                if "place" in locals():
                    list_two_hbonds[k,place] = h_bond
                    del place
                    continue
                else:
                    list_two_hbonds[k,0] = h_bond
            elif h_bond.size==2:
                list_two_hbonds[k,0] = h_bond[0]
                list_two_hbonds[k,1] = h_bond[1]
                continue
            
    print(list_two_hbonds)
    
    for k in range(len(two_hbonds)):
        for l in range(len(two_hbonds)):
            if two_hbonds[k] in list_two_hbonds[l,:]:
                if two_hbonds[l] in list_two_hbonds[k,:]:
                    if "p" in locals():
                        if (two_hbonds[k] and two_hbonds[l]) in p:
                            continue
                        else:
                            molecule_1 = list_two_hbonds[k, list_two_hbonds[k,:]!=two_hbonds[l]]
                            molecule_2 = list_two_hbonds[l, list_two_hbonds[l,:]!=two_hbonds[k]]
                            print(molecule_1, molecule_2)
                            oxygen_1 = grain_input[(grain_input[:,0]=='O') & (grain_input[:,4]==molecule_1)]
                            oxygen_2 = grain_input[(grain_input[:,0]=='O') & (grain_input[:,4]==molecule_2)]
                            d_oxygenoxygen = magnitude_vectors(oxygen_1[:,1].astype(float), oxygen_1[:,2].astype(float), oxygen_1[:,3].astype(float), oxygen_2[:,1].astype(float), oxygen_2[:,2].astype(float), oxygen_2[:,3].astype(float))
                            print(d_oxygenoxygen)
                            if d_oxygenoxygen < D_max_H_bond*coeff_two_hbonds:
                                p = np.append(two_hbonds[k], p)
                                p = np.append(two_hbonds[l], p)
                    else:
                        molecule_1 = list_two_hbonds[k, list_two_hbonds[k,:]!=two_hbonds[l]]
                        molecule_2 = list_two_hbonds[l, list_two_hbonds[l,:]!=two_hbonds[k]]
                        print(molecule_1, molecule_2)
                        oxygen_1 = grain_input[(grain_input[:,0]=='O') & (grain_input[:,4]==molecule_1)]
                        oxygen_2 = grain_input[(grain_input[:,0]=='O') & (grain_input[:,4]==molecule_2)]
                        d_oxygenoxygen = magnitude_vectors(oxygen_1[:,1].astype(float), oxygen_1[:,2].astype(float), oxygen_1[:,3].astype(float), oxygen_2[:,1].astype(float), oxygen_2[:,2].astype(float), oxygen_2[:,3].astype(float))
                        print(d_oxygenoxygen)
                        if d_oxygenoxygen < D_max_H_bond*coeff_two_hbonds:
                            p = two_hbonds[k]
                            p = np.append(two_hbonds[l], p)
    if "p" in locals():
        p, = np.where(np.isin(grain_input[:,4], p))
        print("p " + str(p))
        n = np.append(n,p)
        print("n " + str(n))
    if n.size!=0:
        print("n.size " + str(n.size))
        os.system("mkdir " + str(i) + "-again-" + str(nbr_single_hb))
        for k in range(int(n.size/3)):
            os.system("mv " + str(int(i-k)) + "* ./" + str(i) + "-again-" + str(nbr_single_hb))
        newgrain_file_name = "sans_stickout_mol.xyz"
        newgrain = open(newgrain_file_name,"w")
        print(nbr_atom - n.size, file=newgrain, end="\n \n")
        grain_output = grain_input[:,1:4].astype(float)
        for j in range(nbr_atom):
            if j not in n:
                print(grain_input[j,0] + str("{: {width}.{prec}f}".format(grain_output[j,0], width=25, prec=14)) + str("{: {width}.{prec}f}".format(grain_output[j,1], width=20, prec=14)) + str("{: {width}.{prec}f}".format(grain_output[j,2], width=20, prec=14)), file=newgrain)
        newgrain.close()
        print("Starting GFN2 Geometry optimisation with XTB")
        os.system("xtb " + newgrain_file_name + " --gfn2 --opt > " + newgrain_file_name + "_opt.out")
        os.system("mkdir " + str(i) + "-gfn2-again-" + str(nbr_single_hb))
        os.system("mv xtbopt.log ./" + str(i) + "-gfn2-again-" + str(nbr_single_hb) + "/movie.xyz")
        os.system("cp xtb* wbo charges " + newgrain_file_name + "_opt.out gradient energy sphere.engrad ./" + str(i) + "-gfn2-again-" + str(nbr_single_hb))
        os.system("mv xtbopt.xyz " + newgrain_file_name)
        os.system("rm xtb* wbo charges sphere_opt.out gradient energy sphere.engrad")
        
    
        print("Starting GFN-ff Geometry optimisation with XTB")
        os.system("xtb " + newgrain_file_name + " --verbose --gfnff --opt > " + newgrain_file_name + "_opt.out")
        os.system("mkdir " + str(i) + "-gfnff-again-" + str(nbr_single_hb))
        os.system("rm gfnff_topo")
        os.system("mv " + newgrain_file_name +"_opt.out ./" + str(i) + "-gfnff-again-" + str(nbr_single_hb))
        os.system("mv xtbopt.log ./" + str(i) + "-gfnff-again-" + str(nbr_single_hb) + "/movie.xyz")
        os.system("cp xtbopt.xyz ./" + str(i) + "-gfnff-again-" + str(nbr_single_hb))
        os.system("mv gfnff_charges ./" + str(i) + "-gfnff-again-" + str(nbr_single_hb))
        os.system("mv " + newgrain_file_name + " ./" + str(i) + "-gfnff-again-" + str(nbr_single_hb) + "/input.xyz")
        os.system("mv xtbopt.xyz " + newgrain_file_name)
        atoms_return = io.read(newgrain_file_name)
        os.system("rm " + newgrain_file_name)
        nbr_single_hb = nbr_single_hb + 1
        i = int(i - (n.size)/3)
        print("i " + str(i))
        return atoms_return, i, nbr_final_gfn2, nbr_mol_attrib_problem, nbr_single_hb
    return atoms, i, nbr_final_gfn2, nbr_mol_attrib_problem, nbr_single_hb

def molecule_csv(mol):
    df = pd.read_csv('molecules_reactivity_network.csv', sep='\t')
    atoms = io.read(df.loc[df['species'] == mol, 'pwd_xyz'].values[0]) 
    return atoms

def start_GFN(gfn, input_structure, folder, fixed_core_input):
    GFN_start_time = datetime.now()
    if fixed_core_input is not None:
        subprocess.call(['cp', './' + fixed_core_input, './' + folder])
        process = subprocess.Popen(['xtb', '--input', fixed_core_input, input_structure, '--opt', '--gfn' + gfn, '--verbose'], cwd='./' + folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        process = subprocess.Popen(['xtb', input_structure, '--opt', '--gfn' + gfn, '--verbose'], cwd='./' + folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = open(folder + "/output", "w")
    for line in process.stdout:
        print(line.decode(errors="replace"), end='', file=output)
    stdout, stderr = process.communicate()
    print(stderr.decode(errors="replace"), file=output)
    output.close()
    print('GFN' + str(gfn), str(datetime.now() - GFN_start_time), folder, file=Time_file)

def start_GFN_freq(gfn, input_structure, folder, fixed_core_input):
    if fixed_core_input is not None:
        subprocess.call(['cp', './' + fixed_core_input, './' + folder])
        process = subprocess.Popen(['xtb', '--input', fixed_core_input, input_structure, '--hess', '--gfn' + gfn, '--verbose'], cwd='./'  + folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        process = subprocess.Popen(['xtb', input_structure, '--hess', '--gfn' + gfn, '--verbose'], cwd='./'  + folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = open("./" + folder + "/frequencies", "w")
    for line in process.stdout:
        print(line.decode(errors="replace"), end='', file=output)
    stdout, stderr = process.communicate()
    print(stderr.decode(errors="replace"), file=output)   
    output.close()

def start_GFN_MD(MD_method, input_structure, folder_MD, fixed_core_input):
    process = subprocess.Popen(['xtb', '--input', 'MD.inp', input_structure, '--gfn' + MD_method, '--md', '--verbose'], cwd='./' + folder_MD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = open(folder_MD + "/output", "w")
    for line in process.stdout:
        print(line.decode(errors="replace"), end='', file=output)
    stdout, stderr = process.communicate()
    print(stderr.decode(errors="replace"), file=output)
    output.close()
    subprocess.call(['mv', folder_MD + '/xtb.trj', folder_MD + '/xtb.xyz'])
    io.write('./' + folder_MD + '/xtbmd.xyz', io.read('./' + folder_MD + '/xtb.xyz'))

def start_orca(input_orca, orca_path, folder):
    process = subprocess.Popen([orca_path, input_orca], cwd='./' + folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = open(folder + "/output", "w")
    for line in process.stdout:
        print(line.decode(errors="replace"), end='', file=output)
    stdout, stderr = process.communicate()
    print(stderr.decode(errors="replace"), file=output)
    output.close()

def molecule_selection(list_mol, method_selection):
    if method_selection == 'random':
        list_weight = list_mol[:,1].astype(int)/np.sum(list_mol[:,1].astype(int))
        random_mol = np.random.choice(len(list_mol[:,:]),1, p=list_weight)
        mol = list_mol[random_mol,0][0]
        list_mol[random_mol,1] = str(int(list_mol[random_mol,1]) - 1)
    else:
        for l in range(len(list_mol)):
            if int(list_mol[l,1]) !=0:
                mol = list_mol[l,0]
                list_mol[l,1] = str(int(list_mol[l,1]) - 1)
                break
    return mol, list_mol

if input_file is not None:
    for i in range(len(list_mol)):
        if os.path.isdir('./' + list_mol[i,0] + '_gfn' + gfn + '/') is False:
            mol = list_mol[i,0]
            #make a directory with the name of the molecule + _xtb. To store the xtb files of the molecule to sample
            subprocess.call(['mkdir', mol + '_gfn' + gfn + ''])
            io.write('./' + mol + '_gfn' + gfn + '/' + mol + '_inp.xyz', molecule_csv(mol))
            start_GFN(gfn, mol + '_inp.xyz', mol + '_gfn' + gfn, None)
            subprocess.call(['mv', './' + mol + '_gfn' + gfn + '/xtbopt.xyz', './' + mol + '_gfn' + gfn + '/' + mol + '.xyz'])
            start_GFN_freq(gfn, mol + '.xyz', mol + '_gfn' + gfn + '/', None)

elif os.path.isdir('./' + mol + '_gfn' + gfn + '/') is False:
    #make a directory with the name of the molecule + _xtb. To store the xtb files of the molecule to sample
    subprocess.call(['mkdir', mol + '_gfn' + gfn + ''])
    io.write('./' + mol + '_gfn' + gfn + '/' + mol + '_inp.xyz', molecule_csv(mol))
    start_GFN(gfn, mol + '_inp.xyz', mol + '_gfn' + gfn, None)
    subprocess.call(['mv', './' + mol + '_gfn' + gfn + '/xtbopt.xyz', './' + mol + '_gfn' + gfn + '/' + mol + '.xyz'])
    start_GFN_freq(gfn, mol + '.xyz', mol + '_gfn' + gfn + '/', None)

if restart is not None:
    if MD_method_and_cycle is not None and High_method_and_cycle is not None:
        if (restart-1)%MD_cycle == 0 and (restart-1)%High_cycle==0:
            atoms = io.read('./' + str((restart-1)) + "_high_gfn-" + High_method + '/xtbopt.xyz') 
    elif High_method_and_cycle is not None and MD_method_and_cycle is None:
        if (restart-1)%High_cycle == 0:
            atoms = io.read('./' + str((restart-1)) + "_high_gfn-" + High_method + '/xtbopt.xyz') 
    elif High_method_and_cycle is None and MD_method_and_cycle is not None:
        if (restart-1)%MD_cycle == 0:    
            atoms = io.read('./' + str((restart-1)) + "_MD_" + "GFN-" + MD_method + "_opt" + '/xtbopt.xyz') 
    else:
        atoms = io.read('./' + str(restart - 1) + '/xtbopt.xyz') 
else:
    if "input_building" in globals():
        if input_building is not None:
            if structure is None:
                mol, list_mol = molecule_selection(list_mol, method_selection)
                atoms = io.read('./' + mol + '_gfn' + gfn + '/' + mol + '.xyz')
            else:
                atoms = io.read(structure)
    else:
        if structure is not None:
            atoms = io.read(structure)

            if fixed_structure is True:
                file_xtb_input = open("./fixed_core.inp","w")
                print("$fix", file=file_xtb_input)
                print("    atoms: 1-" + str(len(atoms)), file=file_xtb_input)
                print("$end", file=file_xtb_input)
                file_xtb_input.close()
                fixed_core_input = 'fixed_core.inp'
        else:
            atoms = io.read('./' + mol + '_gfn' + gfn + '/' + mol + '.xyz') 
i = 1
while i < size:
    if restart is not None:
        if i < restart:
            i +=1
            continue
    if "input_building" in globals():    
        if input_building is not None:
            mol, list_mol = molecule_selection(list_mol, method_selection)
    i +=1
    atoms2 = molecule_positioning_simplified(atoms, mol, random_law)
    atoms_old = deepcopy(atoms)
    atoms = atoms + atoms2
    
    folder = str(i)

    subprocess.call(['mkdir', folder])
    io.write('./' + folder + '/cluster.xyz', atoms)

    if i%opt_cycle != 0 and i != size:
        continue

    start_GFN(gfn, 'cluster.xyz', folder, fixed_core_input)
    try:
        atoms = io.read('./' + folder + '/xtbopt.xyz')
        nbr_not_converged = 0 
    except FileNotFoundError:
        print('Error: Not converged')
        subprocess.call(['mv', folder , folder + '-not_converged-' + str(nbr_not_converged)])
        atoms = deepcopy(atoms_old)
        i -= 1
        nbr_not_converged += 1
        if nbr_not_converged > 10: 
            print('Grain building failed after too many converging attempt.')
            exit()
        
        if "input_building" in globals():
            list_mol[np.where(list_mol == mol)[0][0],1] = int(list_mol[np.where(list_mol == mol)[0][0],1]) + 1
        continue

    if MD_method_and_cycle is not None: #Module for start the MD cycles
        if MD_cycle != 0 and i%MD_cycle == 0:
            folder_MD = folder + "_MD_" + "GFN-" + MD_method
            subprocess.call(['mkdir', folder_MD])
            subprocess.call(['cp', 'MD.inp', folder_MD])
            io.write('./' + folder_MD + '/cluster.xyz', atoms)

            start_GFN_MD(MD_method, 'cluster.xyz', folder_MD, fixed_core_input)

            atoms = io.read('./' + folder_MD + '/xtbmd.xyz')

            folder_MD_opt = folder_MD + "_opt"
            subprocess.call(['mkdir', folder_MD_opt])

            io.write('./' + folder_MD_opt + '/cluster.xyz', atoms)

            start_GFN(gfn, 'cluster.xyz', folder_MD_opt, fixed_core_input)

            atoms = io.read('./' + folder_MD_opt + '/xtbopt.xyz') 

    if High_method_and_cycle is not None: # Module for starting the High level relaxation cycle 
        if gfn == 'ff' and i%High_cycle != 0 and i == 50 and restart is None and High_cycle > 50:
            folder_High = folder + "_high_gfn-" + High_method
            subprocess.call(['mkdir', folder_High])
            io.write('./' + folder_High + '/cluster.xyz', atoms)

            start_GFN(High_method, 'cluster.xyz', folder_High, fixed_core_input)

            atoms = io.read('./' + folder_High + '/xtbopt.xyz') 

        elif i%High_cycle == 0:
            folder_High = folder + "_high_gfn-" + High_method
            subprocess.call(['mkdir', folder_High])
            io.write('./' + folder_High + '/cluster.xyz', atoms)

            start_GFN(High_method, 'cluster.xyz', folder_High, fixed_core_input)

            atoms = io.read('./' + folder_High + '/xtbopt.xyz') 

    if final_gfn2 is True and i == size and gfn !='2':
        folder_final_gfn2 = folder + '-final_gfn2'
        subprocess.call(['mkdir', folder_final_gfn2])
        io.write('./' + folder_final_gfn2 + '/cluster.xyz', atoms)
        start_GFN('2', 'cluster.xyz', folder_final_gfn2, fixed_core_input)
        atoms = io.read('./' + folder_final_gfn2 + '/xtbopt.xyz')

#    if final_gfn2 is True and i == size and gfn !='2':
#        if MD_method_and_cycle is not None: #Module for start the MD cycles
#            if MD_cycle != 0 and i%MD_cycle != 0:
#                folder_final_gfn2 = folder + '-final_gfn2'
#                subprocess.call(['mkdir', folder_final_gfn2])
#                io.write('./' + folder_final_gfn2 + '/cluster.xyz', atoms)
#
#                start_GFN('2', 'cluster.xyz', folder_final_gfn2)
#
#                atoms = io.read('./' + folder_final_gfn2 + '/xtbopt.xyz')

    if i == size and check_surface == True: #Module that checks the grains surfaces for oddities
        print('check surface')
        atoms, i, nbr_final_gfn2, nbr_mol_attrib_problem, nbr_single_hb = check_surface_agermain2021(atoms, i, size, nbr_final_gfn2, nbr_mol_attrib_problem, nbr_single_hb)


    #print("i fin boucle " + str(i))
    #print("size " + str(size))

if input_file is not None:
    if input_orca is not None:
        folder_orca = folder + '-final_orca'
        subprocess.call(['mkdir', folder_orca])
        io.write('./' + folder_orca + '/cluster.xyz', atoms)
        file_orca_inp = open('./' + folder_orca + '/input_orca.inp', 'w')
        print('!' + input_orca[0,1] + ' opt \n%pal \nnproc=' + input_orca[2,1] + ' \nend \n* xyzfile 0 1 cluster.xyz END',file=file_orca_inp)
        file_orca_inp.close()
        start_orca('input_orca.inp',input_orca[1,1], folder_orca)
        atoms = io.read('./' + folder_orca + '/input_orca.xyz')

if final_gfn2 == True:
    io.write('grain_' + str(size) + '_gfn' + str(gfn) + '_cycle-' + str(opt_cycle) + '_' + random_law + '_gfn2.xyz', atoms)
else:
    io.write('grain_' + str(size) + '_gfn' + str(gfn) + '_cycle-' + str(opt_cycle) + '_' + random_law + '.xyz', atoms)

print('Execution time:', str(datetime.now() - startTime), file=Time_file)
Time_file.close()
