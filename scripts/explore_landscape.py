"""


for i in {1..10}; do nice python explore_landscape.py -i $i & done

for i in {20..300..8}; do nice python explore_landscape.py -i $i ;  done
for i in {21..300..8}; do nice python explore_landscape.py -i $i ;  done
for i in {22..300..8}; do nice python explore_landscape.py -i $i ;  done
for i in {23..300..8}; do nice python explore_landscape.py -i $i ;  done
for i in {24..300..8}; do nice python explore_landscape.py -i $i ;  done
for i in {25..300..8}; do nice python explore_landscape.py -i $i ;  done
for i in {26..300..8}; do nice python explore_landscape.py -i $i ;  done
for i in {27..300..8}; do nice python explore_landscape.py -i $i ;  done

# traces
for i in {3000..3010}; do nice python explore_landscape.py -i $i -t & done

# for even samples of latent space
awk 'BEGIN{i=0} {print "nice python explore_landscape.py  -i", i++, "-s", $1, "-t"}' \
        ../DHFR/plot/sample_latent_points_seqs.txt | split -l 200 
for file in xa?; do cat $file | sh & done
rm -f xa?

"""
import numpy as np
import torch

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

energy_calc(np.random.randint(20, size=config.L)) # array(1.66720488)

# for amino acid index i we have a list of all the other amino acids indices
# this makes it easy to pick an actual mutation and make sure we are not picking the 
# same amino acid
mut_l = [[j for j in range(config.AA_L) if j != i] for i in range(config.AA_L)]


def mutate_protein(single_mutant_idx, prot):
    """
        * single_mutant_idx * : an integer from the array single_mutant_indexer
                                (integer in the range 0 -> 186 * 19)
        * prot *              : any protein as a numpy array
                                (integer array of length 186 with vals between 
                                 0 and 19 (inclusive))
    """
    mut = prot.copy()
    # which mutant index is selected from the mut_l array
    mut_aa_idx = single_mutant_idx % (config.AA_L - 1)
    # which site is mutated
    mut_site_idx = single_mutant_idx // (config.AA_L - 1)
    # first find out what site is mutated in prot
    # and then selected the mut_aa_idx value to find out what it mutates to
    new_aa = mut_l[prot[mut_site_idx]][mut_aa_idx]
    mut[mut_site_idx] = new_aa
    return mut

# testing function mutate_protein
np.random.seed(0)
test_start_prot = np.random.randint(config.AA_L, size=config.L)

test_mutated_prot = mutate_protein(1000, test_start_prot)

# site 52 (== 1000//19) should be the only site mutated
assert(np.where(test_mutated_prot != test_start_prot)[0].item() == 1000 // 19)

# what it should have been mutated to
assert(mut_l[test_start_prot[52]][1000 % 19] == test_mutated_prot[52])

# assert site 52 is different
assert(test_start_prot[52] != test_mutated_prot[52])

num_single_mutants = (config.AA_L - 1) * config.L

# create an index list of 19*186 mutants
single_mutant_indexer = np.arange(num_single_mutants)

def calc_dist(p1, p2=WT):
    return int(sum(p1 != p2))

# save results to output json
output_dir = f"{config.WORKING_DIR}/adaptive_walk/even"

def get_trace_filename(start_seed_offset):
    return f"{output_dir}/walk_{start_seed_offset}.csv" 

def write_trace(start_seed_offset, text):
    filename = get_trace_filename(start_seed_offset)
    with open(filename, "a") as fh_out: # append
        print(text, file=fh_out)

if __name__ == "__main__":
    import time
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index_offset",
                    type=int,
                    help="offset to random seed",
                    required=True)
    parser.add_argument("-t", "--trace",
                    default=False,
                    action="store_true")
    parser.add_argument("-s", "--sequence",
                    default="",
                    required=False)


    args = parser.parse_args()
    start_seed_offset = args.index_offset
    trace = args.trace

    np.random.seed(100 + start_seed_offset)
    start_protein = None
    if args.sequence:
        start_protein = np.array(list(map(config.AA_MAP.get, args.sequence)), 
                                    dtype=int)
    else:
        start_protein = np.random.randint(config.AA_L, size=config.L)
    start_energy = energy_calc(start_protein)
    current_protein = start_protein
    current_energy = start_energy

    print(f"Starting mutant : {current_energy:.4f} ", 
          f"wt-distance : {calc_dist(current_protein):03d}")
    if trace:
        with open(get_trace_filename(start_seed_offset), "wt") as fh_out:
            print("protein,energy,dist_wt", file=fh_out)
        write_trace(start_seed_offset,
                f"{config.prot_to_string(current_protein)}," \
                f"{current_energy.item()},{calc_dist(current_protein)}")
            

    while True:
        np.random.shuffle(single_mutant_indexer)
        found_better = False
        for single_mutant_idx in single_mutant_indexer:
            mutant_protein = mutate_protein(single_mutant_idx, current_protein)
            mutant_energy = energy_calc(mutant_protein)
            if mutant_energy < current_energy:
                current_protein = mutant_protein
                current_energy = mutant_energy
                found_better = True
                print(f"Found Better mutant : {current_energy:.4f} "
                      f"wt-distance : {calc_dist(current_protein):03d}")
                if trace:
                    write_trace(start_seed_offset,
                        f"{config.prot_to_string(current_protein)}," \
                        f"{current_energy.item()},{calc_dist(current_protein)}")
                break
        if found_better is False:
            # we didn't find anything better. Terminate the search
            print("No more mutants found")
            break


    results = {'start_protein':config.prot_to_string(start_protein),
               'start_energy':start_energy.item(),
               'start_dist_wt':calc_dist(start_protein),
               'end_protein':config.prot_to_string(current_protein),
               'end_energy':current_energy.item(),
               'end_dist_wt': calc_dist(current_protein)}

    with open(f"{output_dir}/walk_{start_seed_offset}.json", "w", 
                                            encoding="utf-8") as outf:
        json.dump(results, outf, ensure_ascii=True, indent=4)


