import numpy as np
import energy

class EnergyFunctionCalculator:
    """ Calculate Energy for integer index arrays via __call__
        Can pass in either a single protein or an MSA 
    """

    def __init__(self, 
                 h_i_a, # local fields
                 e_i_a_j_b): # pairwise couplings
        self.h_i_a = h_i_a
        self.e_i_a_j_b = e_i_a_j_b
        self.L = self.h_i_a.shape[0]

    def __call__(self, protein, *args, **kwargs):
        """ We can pass in an array of numpy ints representing 
            a protein sequence or an MSA. 
            Array shape (L, ) gives an energy for that sequence
            Array shape (n_seq, L) returns an energy array of length
                n_seq for that MSA
        """
        shape = protein.shape
        ret = None
        if len(shape) == 1:
            if shape[0] != self.L:
                raise ValueError(f"Single Protein needs length {self.L}"
                                 f" Shape : {shape}")
            else:
                ret = energy_calc_single(protein, self.h_i_a, 
                        self.e_i_a_j_b)
        elif len(shape) == 2:
            if shape[1] != self.L:
                raise ValueError(f"MSA Protein needs {self.L} in 2nd dim."
                                 f" Shape : {shape}")
            else:
                ret = energy_calc_msa(protein, self.h_i_a, 
                        self.e_i_a_j_b)
        else:
            raise ValueError(f"input can only be dimension 1 or 2."
                             f" Shape : {shape}")
        return ret

 
def energy_calc_single(prot_np, h_i_a, e_i_a_j_b):
    """prot_np  : A numpy integer array of length L 
                  (with the correct Amino acid index in each position)
                  Maximum value prot_np should be 20
       h_i_a    : field values (shape = (L, q))
       e_i_a_j_b: coupling values (shape = (L, q, L, q))

       Returns : energy (float)
    """
    energy = energy_calc_msa(prot_np[np.newaxis, ...], h_i_a, e_i_a_j_b)
    return energy.squeeze()


def energy_calc_msa(msa, h_i_a, e_i_a_j_b):
    """msa     : A numpy integer array of shape (nseqs, L) 
                  (with the correct Amino acid index in each position)
                  Maximum value prot_np should be 20 (or q)
       h_i_a    : field values (shape = (L, q))
       e_i_a_j_b: coupling values (shape = (L, q, L, q))

       NOTE: this function is supposed to be a faster version of 
       energy_calc.energy_calc_msa. It does no real bounds checking so 
       we do some rudimentary checks before passing to C++
        
       Returns: energy (np float array of size nseqs)
    """

    if msa.max() >= h_i_a.shape[1]: # bounds checking
        raise ValueError("Max value in msa should be less than alphabet size")
    return energy.energy_calc_msa(msa, h_i_a, e_i_a_j_b)


def energy_calc_single_mutants(seq, h_i_a, e_i_a_j_b):
    """seq      : A numpy integer array of shape (L,) 
                  (with the correct Amino acid index in each position)
                  Maximum value prot_np should be 20 (or q)
       h_i_a    : field values (shape = (L, q))
       e_i_a_j_b: coupling values (shape = (L, q, L, q))
       return a tuple (All single mutants, energies)
    """
    if seq.max() >= h_i_a.shape[1]: # bounds checking
        raise ValueError("Max value in seq should be less than alphabet size")
    return energy.energy_calc_single_mutants(seq.squeeze(), h_i_a, e_i_a_j_b)

def create_single_mutant(i, a, wt):
    mut = wt.copy()
    mut[i] = a
    return (mut)


if __name__ == "__main__":
    import time
    import torch
    import config

    model_prefix = "DHFR_kit_taq"
    h_i_a = torch.load(f"../working/{model_prefix}_h_i_a.pt").numpy()
    e_i_a_j_b = torch.load(f"../working/{model_prefix}_e_i_a_j_b.pt").numpy()

    energy_calc = EnergyFunctionCalculator(h_i_a, e_i_a_j_b)

    WT_PROT_NP = config.WT_AA_TENSOR.numpy()
    print(f"WT Energy: {energy_calc(WT_PROT_NP):.2f}")

    random_prot = (WT_PROT_NP + 5) % 20
    print(f"Random Energy: {energy_calc(random_prot):.2f}")
    small_msa = np.stack([WT_PROT_NP, random_prot])
    msa = small_msa

    #big_msa = np.repeat(WT_PROT_NP[np.newaxis, :], 100000, axis=0)
    #msa = big_msa

    print(f"h_i_a: shape={h_i_a.shape} dtype={h_i_a.dtype} \n{h_i_a.flags}")
    print(f"e_i_a_j_b: shape={e_i_a_j_b.shape} dtype={e_i_a_j_b.dtype}"
          f"\n{e_i_a_j_b.flags}")
    print(f"msa: shape={msa.shape} dtype={msa.dtype} \n{msa.flags}")

    start_time = time.time()
 
    results = energy_calc(msa)
    print(f"results.shape={results.shape}\n{results}")

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print()
    print('Checking single mutant calculations')
    start_time = time.time()
    single_mutants, single_energies = energy.energy_calc_single_mutants(
            WT_PROT_NP.squeeze(), h_i_a, e_i_a_j_b)

    # Now create a single mutant MSA and use the energy_calc_msa function
    # to compute energy
    single_mutant_msa = np.array([
            create_single_mutant(i=m[0], a=m[1], wt=WT_PROT_NP.squeeze()) 
            for m in single_mutants])
    single_mutant_msa_energies = energy_calc_msa(single_mutant_msa, h_i_a,
            e_i_a_j_b)

    max_abs_diff = np.abs(single_energies - single_mutant_msa_energies).max()
    print(f"Max difference in single mutant energies calculated two ways "
          f"{max_abs_diff:4E}")
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

