import numpy as np
from tqdm import tqdm


import config
import energy_py

@config.memory.cache
def find_best_single_double_mutant(starting_seq, h_i_a, e_i_a_j_b, 
                                    verbose=True,
                                    WT = config.WT_AA_TENSOR.numpy() ):
    """ Find fittest single or double mutant from starting sequence
        taking care to make sure that we move "away" from wild-type
        (We do not allow any back substitutions towards wild-type)
    """
    
    dist_threshold = config.calc_min_dist(starting_seq, WT)
    min_energy = energy_py.energy_calc_single(starting_seq, h_i_a, e_i_a_j_b)
    min_mutant = []
    
    single_mutants, single_energies = energy_py.energy_calc_single_mutants(
            starting_seq, h_i_a, e_i_a_j_b)
    
    if verbose: print("Searching single mutants")
    # find best single mutant
    allowed_mutants = (WT[single_mutants[:, 0]] != single_mutants[:, 1])
    if np.any(single_energies[allowed_mutants] < min_energy):
        if verbose: 
            print("Found fitter single mutant increasing distance from WT")
        min_energy_idx = single_energies[allowed_mutants].argmin()
        min_energy = single_energies[allowed_mutants][min_energy_idx]
        min_mutant = [single_mutants[allowed_mutants][min_energy_idx, :].copy()]
        print(str(min_mutant))
      
    if verbose: print("Searching double mutants")
    # now search in single mutants of single mutants (i.e. double mutants)
    for m in tqdm(single_mutants):
        if WT[m[0]] == m[1]: continue
        mut = energy_py.create_single_mutant(i=m[0], a=m[1], 
                wt=starting_seq.squeeze())
        double_mutants, double_energies = energy_py.energy_calc_single_mutants(
                                                mut, h_i_a, e_i_a_j_b)
        allowed_mutants = (WT[double_mutants[:, 0]] != double_mutants[:, 1])
        if np.any(double_energies[allowed_mutants] < min_energy):
            if verbose:
                print("Found fitter double mutant increasing distance from WT")
            min_energy_idx = double_energies[allowed_mutants].argmin()
            min_energy = double_energies[allowed_mutants][min_energy_idx]
            min_mutant = [m.copy(), double_mutants[allowed_mutants][
                                min_energy_idx, :].copy()]
            print(str(min_mutant))

    # convert best mutant back into sequence space
    min_seq = starting_seq.copy()
    if len(min_mutant) != 0:
        for m in min_mutant:
            min_seq[m[0]] = m[1]
    return min_seq, min_energy


if __name__ == "__main__":
    pass
