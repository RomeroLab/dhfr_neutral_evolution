"""Functions to calculate frequency counts"""

import gzip

import numpy as np
import pandas as pd
import torch

import config

def map_dict(lam, d):
    return {k:lam(d[k]) for k in d}

def convert_to_one_hot(int_seqs, q):
    return torch.eye(q, dtype=torch.int32)[int_seqs]

def calc_pairwise_counts(msa_one_hot):
    return torch.einsum("nlq,nkp->lqkp", msa_one_hot, msa_one_hot)


@config.memory.cache
def calc_mutant_counts(msa_nts_filename, max_seq_length=0, 
                        interaction=True, cpp=True):
    """
        Returns main and interaction frequency counts

        `max_seq_length` : Only read this number of sequences (for testing)
                            If it is zero or negative it returns all seqs
        `interaction` : Only return main effect terms
        `cpp` : trades compute for memory and is generally faster
    """
    msa = config.get_codon_msa_as_int_array(msa_nts_filename) # (N, L) int
    if max_seq_length > 0: # reduce the size of the MSAs for testing purposes
        msa = msa[:, :max_seq_length]
    counts_main, counts_int = None, None
    if cpp:
        from paircounts import calc_paircounts as cpc
        counts_int = torch.from_numpy(cpc(msa.numpy(), q=config.qc))
        # get counts_main from the diagonal of the interaction counts
        runner = torch.arange(msa.shape[1]) # index along length of protein
        counts_main = torch.diagonal(counts_int[runner, :, runner, :], 
                                        dim1=-2, dim2=-1) # batch diagonal
        if not interaction:
            counts_int = None
    else:
        # .long() increases memory but torch wants a LongTensor to index
        msa_one_hot = convert_to_one_hot(msa.long(), 
                                            q=config.qc) # (N, L, qc) int
        counts_main = msa_one_hot.sum(axis=0) # (L, qc) long array
        if interaction: # (L, qc, L, qc) long 
            counts_int = calc_pairwise_counts(msa_one_hot) 
    return counts_main, counts_int

def undo_joblib_memory():
    global calc_mutant_counts
    calc_mutant_counts = calc_mutant_counts.__wrapped__

def calc_tensor_size(x, human=True):
    size = x.element_size() * x.numel()
    if human:
        if size > 500e6:
            size = f"{size / 1e9:.2f}G"
        else:
            size = f"{size / 1e6:.2f}M"
    return size

def create_codon_transition_matrix(nucleotide_trans_mat):
    """ Calculate codon transition matrix from nucleotide transition matrix
        
        Assumption: We assume the nucleotide_trans_mat has initial states as
            columns and transitioned states as rows. And the return argument
            shares that assumption. However, this function should work
            transparently. i.e.  if we have the initial states as rows and the
            transitioned states as columns then the same will be true for the
            return argument.  
        Args:
            nucleotide_trans_mat:  a nucleotide transition matrix (4x4) 
                                   (as pandas dataframe)
        Return:
            codon_trans_mat: a codon transition matrix (64x64)(
                             (as pandas dataframe)
    """
    # 6 dim Cartesian product of A,C,T,G
    nucleotide_idxs = np.meshgrid(*([nucleotide_trans_mat.index] * 6))
    n1, n2, n3, n4, n5, n6 = list(map(lambda x: x.flatten(), nucleotide_idxs))
    codons_from = n1 + n2 + n3 # AAA, AAC, ...
    codons_to = n4 + n5 + n6 
    # Calculate transition probability by multiplying individual transitions
    codon_trans = nucleotide_trans_mat.lookup(n4, n1) * \
                    nucleotide_trans_mat.lookup(n5, n2) * \
                    nucleotide_trans_mat.lookup(n6, n3)
    codon_trans = pd.DataFrame(data={'prob':codon_trans, 'to':codons_to,
                                    'from':codons_from})
    # Reshape long to wide to create a lookup table
    codon_trans_mat = codon_trans.pivot(index="to", columns="from", 
                                            values="prob")
    return codon_trans_mat

def create_codon_trans_mat_from_csv(csv_filename=config.NT_TRANS_MAT_CSV):
    """ Order the codons in the transition matrix and exclude stops
        This will give us a matrix of size qc X qc  
    """
    # one round nucleotide transition matrix
    nt_trans_mat = pd.read_csv(csv_filename, index_col=0) # shape (4, 4)
    # convert to codon transition matrix
    codon_trans_mat = create_codon_transition_matrix(nt_trans_mat)

    # create indexing for rows and columns of transition matrix
    # in the same order as the codon_map
    codon_rows = np.array([i for i, c in enumerate(codon_trans_mat.index)
                                if c in config.CODON_MAP])
    codon_cols = np.array([i for i, c in enumerate(codon_trans_mat.columns)
                                if c in config.CODON_MAP])
    
    # subset transition map to codon transition matrix excluding stop codons
    # Probability of transition goes from columns to rows. Shape (qc, qc)
    codon_ex_stops_trans_mat = codon_trans_mat.iloc[codon_rows,
                                                    codon_cols].to_numpy()
    return codon_ex_stops_trans_mat 
 

def load_torch_tensor(filename, requires_grad=False):
    """ load a gzipped or regular torch tensor depending on filename"""
    opener = open
    if filename.endswith(".gz"):
        opener = gzip.open
    ret = None
    with opener(filename, "rb") as fh:
        ret = torch.load(fh)
        ret.requires_grad = requires_grad
    return ret

def calc_energy_pytorch(seq, h, e, q=20):
    seq_one_hot = torch.eye(q)[seq] # make one_hot
    L = seq_one_hot.shape[0]
    # half the interaction terms as we are going to sum over all i,j
    # and not just i < j
    e_half = e.clone() / 2 
    # also zero out the diagonal so we don't add diagonal interaction terms if
    # there are any
    e_half[range(L), :, range(L), :] = 0 
    energy = torch.einsum("ia,ia", seq_one_hot, h.float()) + \
             torch.einsum("ia,iajb,jb", seq_one_hot, e_half.float(), 
                                        seq_one_hot)
    return -energy

if __name__ == "__main__":
    pass
