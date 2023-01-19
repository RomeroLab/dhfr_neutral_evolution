""" Find shortest path from WT to most common sequence in Round 15"""
import math
import itertools

import numpy as np
import pandas as pd

import torch

import json

import config
import utils

from energy_py import EnergyFunctionCalculator, energy_calc_msa, energy_calc_single
import energy_py

WT = config.WT_AA_TENSOR.numpy()


model_prefix = "DHFR_incl_main_kit_taq_mse"


h_i_a = torch.load(f"../working/{model_prefix}_h_i_a.pt").numpy()
e_i_a_j_b = torch.load(f"../working/{model_prefix}_e_i_a_j_b.pt").numpy()
#e_i_a_j_b = e_i_a_j_b / 2.

energy_calc = EnergyFunctionCalculator(h_i_a, e_i_a_j_b)

WT_energy = energy_calc(WT) # lower is fitter
WT_energy # array(0.7909182)


designed_proteins = pd.read_csv(f"{config.WORKING_DIR}/{model_prefix}_designed.csv")
start_designed_protein = designed_proteins.seq.iloc[0]
terminal_designed_protein = designed_proteins.seq.iloc[-1]

with open(f"{config.WORKING_DIR}/adaptive_walk/even/walk_0.json") as fh:
    x = json.load(fh)
    convergent_protein = x["end_protein"]

def print_prot_diff(s1, s2):
    for i, (ss1, ss2) in enumerate(zip(s1, s2)):
        if ss1 != ss2:
            print(f"{i:03d}: {ss1}->{ss2}")

print("Where does terminal_designed_protein differ from convergent_protein")
print_prot_diff(terminal_designed_protein, convergent_protein)
print()
# 007: V->A
# 090: K->E
# 131: M->R
# 166: S->G


print("Where does start_designed_protein differ from WT")
print_prot_diff(config.WT_AA, start_designed_protein)
print()
# 018: N->D
# 040: S->G
# 079: K->I
# 094: D->G
# 141: F->L
# 143: S->C
# 144: D->N
# 165: L->P
# 179: E->A
# 185: D->V

mutants = pd.DataFrame([(i, x1, x2) for i, (x1, x2) in 
                enumerate(zip(WT, start_designed_protein_np)) if x1 != x2], 
                columns=["idx","aa_wt","aa_start"])

num_muts = len(mutants)

muts_d = {i:np.array(list(itertools.combinations(range(num_muts), i))) 
                for i in range(1,num_muts)}

def calc_energy_mut(mut_idxs):
    """mut is 1d-numpy integer array with values between 0 
       and (num_muts - 1). We look up the actual mutants in the 
       mutants dataframe and apply them and return the energy
    """
    m = WT.copy()
    m[mutants.idx.iloc[mut_idxs]] = mutants.aa_start.iloc[mut_idxs]
    return energy_calc(m).item()

@config.memory.cache
def get_mutants_energy_dict(muts_d):
    return {key:np.apply_along_axis(calc_energy_mut, axis=1, arr=arr) 
            for key, arr in muts_d.items()}


energy_d = get_mutants_energy_dict(muts_d)


def get_one_mutant_adjacency(arr_num_muts, arr_num_muts_plus_one):
    """ Find out which n mutants are only 1 mutant difference from the 
        n+1 mutants 
        Ex. Not all 3 mutants can mutate to all 4 mutants so we have a flag 
            between them if they can be reached by 1 mutant (adjaceny matrix)
    """
    assert(arr_num_muts_plus_one.shape[1] == arr_num_muts.shape[1] + 1)
    ret = np.zeros((arr_num_muts.shape[0], arr_num_muts_plus_one.shape[0]), dtype=int)
    for ix, x in enumerate(arr_num_muts):
        x = set(x)
        for iy, y in enumerate(arr_num_muts_plus_one):
            if x.issubset(y):
                ret[ix, iy] = 1
    return ret

adjs_d = {i:get_one_mutant_adjacency(muts_d[i], muts_d[i+1]) 
                    for i in range(1, num_muts-1)}


energy_adjs_d = {}
for i, arr in adjs_d.items():
    ix, iy = np.where(arr) # where the single mutant transitions can happen
    energy_adjs_d[i] = (ix, iy, energy_d[i][ix] + energy_d[i+1][iy])

# count all the paths that we can travel and sum their energy
ix = {}
iy = {}
path_energy = {}

#ix[1], iy[1] = np.where(adjs_d[1])
#path_energy[2] = energy_d[1][ix[1]] + energy_d[2][iy[1]]
#
#ix[2], iy[2] = np.where(adjs_d[2][iy[1], :])
#path_energy[3] = path_energy[2][ix[2]] + energy_d[3][iy[2]]
#...

for i in range(1, num_muts-1):
    if (i == 1):
        ix[i], iy[i] = np.where(adjs_d[i])
        path_energy[i+1] = energy_d[i][ix[i]] + energy_d[i+1][iy[i]]
    else:
        ix[i], iy[i] = np.where(adjs_d[i][iy[i-1]])
        path_energy[i+1] = path_energy[i][ix[i]] + energy_d[i+1][iy[i]]
    print(path_energy[i+1].shape)


assert(path_energy[num_muts-1].size == math.factorial(num_muts))

shortest_path_idx = path_energy[9].argmin().item()
shortest_path_energy = path_energy[9][shortest_path_idx]

print(muts_d[9][iy[8][shortest_path_idx]])
#[0 1 2 3 5 6 7 8 9]
sp8 = ix[8][shortest_path_idx]
print(muts_d[8][iy[7][sp8]])
sp7 = ix[7][sp8]
print(muts_d[7][iy[6][sp7]])
sp6 = ix[6][sp7]
print(muts_d[6][iy[5][sp6]])
sp5 = ix[5][sp6]
print(muts_d[5][iy[4][sp5]])
sp4 = ix[4][sp5]
print(muts_d[4][iy[3][sp4]])
sp3 = ix[3][sp4]
print(muts_d[3][iy[2][sp3]])
sp2 = ix[2][sp3]
print(muts_d[2][iy[1][sp2]])
sp1 = ix[1][sp2]
print(sp1)

# loop form of the above

def mut_to_string(mut_idxs):
    """mut is 1d-numpy integer array with values between 0 
       and (num_muts - 1). We look up the actual mutants in the 
       mutants dataframe and apply them and return the energy
    """
    m = WT.copy()
    m[mutants.idx.iloc[mut_idxs]] = mutants.aa_start.iloc[mut_idxs]
    return config.prot_to_string(m)

protein_path = {}
protein_path[10] = start_designed_protein

sp = {} # shortest path indices
for i in reversed(range(1, num_muts)):
    if i == 9:
        sp[9] = shortest_path_idx
    else:
        sp[i] = ix[i][sp[i+1]]
    if i == 1:
        p = muts_d[i][sp[i]]
    else:
        p = muts_d[i][iy[i-1][sp[i]]] 
    print(p)
    protein_path[i] = mut_to_string(p)

protein_path_pd = pd.DataFrame.from_dict(protein_path, orient="index", 
                                columns=["protein"]).sort_index().reset_index()

protein_path_pd.to_csv(f"{config.WORKING_DIR}/{model_prefix}_shortest_path.csv", 
        index=False)


for _ in range(10): # create random paths and check their energy
    random_path = np.random.choice(num_muts, replace=False, size=num_muts)
    random_path_energy = 0.
    for i in range(1, num_muts):
        random_path_energy += calc_energy_mut(random_path[:i])
    print(random_path_energy)
    assert(random_path_energy >= shortest_path_energy)




