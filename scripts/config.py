""" Configuration variables and utilties for scripts """

import itertools
import gzip
import pathlib

import numpy as np
import torch
import Bio
import Bio.Seq

import joblib

WORKING_DIR="../working"
ASSETS_DIR="../working/assets"

JOBLIB_CACHE_DIR=WORKING_DIR
memory = joblib.Memory(JOBLIB_CACHE_DIR, verbose=0)

def disable_joblib_memory():
    global memory
    memory = joblib.Memory(None, verbose=0)

PRECISION = torch.float32
EPSILON = 0 # Set this to a small number if need to prevent log(0)

# Using the WT_DNA reference string (no stop or start codon)
WT_DNA = "GTTCGACCATTGAATTGTATTGTAGCAGTATCACAAAATATGGGTATTGG" \
         "TAAAAATGGTGATTTGCCATGGCCACCTCTACGTAATGAATTCAAGTATT" \
         "TTCAGCGAATGACTACTACTAGTTCAGTTGAAGGTAAACAAAATTTAGTT" \
         "ATTATGGGTCGTAAAACATGGTTTAGTATCCCTGAAAAAAATCGACCATT" \
         "AAAAGATCGAATCAATATTGTTTTAAGTCGTGAATTAAAAGAACCTCCAC" \
         "GAGGTGCTCATTTTTTAGCTAAAAGTTTAGATGATGCATTGCGACTAATC" \
         "GAACAACCAGAATTGGCATCAAAAGTTGATATGGTATGGATTGTAGGTGG" \
         "TAGTTCAGTTTATCAGGAAGCAATGAATCAACCTGGTCACTTACGATTGT" \
         "TTGTTACTCGAATCATGCAGGAATTTGAAAGTGATACTTTTTTTCCAGAA" \
         "ATTGATTTGGGTAAATATAAATTACTACCTGAATATCCAGGTGTTCTAAG" \
         "TGAAGTTCAGGAAGAAAAAGGTATCAAATATAAATTTGAAGTTTATGAGA" \
         "AGAAAGAT"

WT_AA = "VRPLNCIVAVSQNMGIGKNGDLPWPPLRNEFKYFQRMTTTSSVEGKQNLVI" \
        "MGRKTWFSIPEKNRPLKDRINIVLSRELKEPPRGAHFLAKSLDDALRLIEQ" \
        "PELASKVDMVWIVGGSSVYQEAMNQPGHLRLFVTRIMQEFESDTFFPEIDL" \
        "GKYKLLPEYPGVLSEVQEEKGIKYKFEVYEKKD"

assert str(Bio.Seq.Seq(WT_DNA).translate()) == WT_AA, \
        "WT_AA is not the correct translation of WT_DNA"

# Length of the Protein Sequence
L = len(WT_AA)

## Amino acid and codon mappings 
# (strings to numbers, codons to amino acids etc)
AA_ALPHABET="RKDEQNHSTCYWAILMFVPG-"
AMINO_ACIDS = np.array([aa for aa in AA_ALPHABET], "S1")

AAs_string = AA_ALPHABET.replace("-", "") # drop the gap character
AAs = np.array([aa for aa in AAs_string], "S1")
AA_L = AAs.size # alphabet size
AA_MAP = {a:idx for idx, a in enumerate(AAs_string)}
# Maps integer index back to amino acids
INV_AA_MAP = {v:k for k, v in AA_MAP.items()} 

# create a mapping for codons that are not stop codons
# This will be used for one-hot encoding the sequences
codon_table = Bio.Data.CodonTable.standard_dna_table
CODON_MAP = {c:i for i, c in enumerate(
                            sorted(codon_table.forward_table.keys()))}

qa = len(AA_MAP) # qa = 20 (20 amino acids)
qc = len(CODON_MAP) # qc = 61 (all codons excluding stop codons)

#CODON_AA_MAP
CODON_AA_MAP = torch.LongTensor([AA_MAP[codon_table.forward_table[c]] for 
                                  c, idx in CODON_MAP.items()])
# Binary matrix that translates between Codon index and Amino Acid index
#       shape (qc, qa) binary
def calc_codon_aa_matrix(): #function incase we need to recreate this matrix
    return torch.eye(len(AA_MAP), dtype=PRECISION)[CODON_AA_MAP] 
CODON_AA_MATRIX = calc_codon_aa_matrix()

def codon_seq_to_int_list(seq): 
    return [CODON_MAP[seq[3*i:(3*i+3)]] for i in range(len(seq)//3)]

## WildType as an integer array
WT = torch.LongTensor(codon_seq_to_int_list(WT_DNA))
WT_AA_TENSOR = CODON_AA_MAP[WT]

## Contact Map
CONTACT_MAP_MATRIX_FILENAME = "../DHFR/contact_map.npy"

## Multiple Sequence Alignment files
DATADIR_EARLY = pathlib.Path("../DHFR/GenEarly/Oct10_QComp")
DATADIR_15 = pathlib.Path("../DHFR/Gen15")

# Order MSA files by sequencing data
msa_filesd_raw = {i:(DATADIR_EARLY/f"Round{i}_Q15_C10_nts.aln.gz") for
                    i in range(1, 5+1)}
msa_filesd_raw[15] = DATADIR_15 / "Gen15_nts.aln.gz"

MSA_FILESd = msa_filesd_raw.copy()

# Nucleotide Transition matrix csv file
NT_TRANS_MAT_CSV = "../DHFR/unsorted_transition/pd_transition_round_1_kit_taq.csv"

## Functions to read Multiple Sequence Alignment files
def get_msa_from_aln_iter(aln_filename, size_limit=None):
    """Reads a (plain text) aln file (can be gzipped also) and iterate
        over string sequences 
    Args:
        aln_filename    : Filename or filename of ALN file to read
        size_limit      : Return upto size_limit sequences
    """
    opener = open
    if str(aln_filename).endswith(".gz"):
        opener = gzip.open
    with opener(aln_filename, "rt") as fh:
        seq_io_gen = (line.strip() for line in fh)
        # Read only size_limit elements of the generator
        # if size_limit is None then we will read in everything
        seq_io_gen_slice = itertools.islice(seq_io_gen, size_limit) 
        # We need the yield from statement below so that the file
        # handle is kept open as long as we are reading from it
        # returning a generator expression would close fh
        yield from (seq.upper() for seq in seq_io_gen_slice)

def get_codon_msa_as_int_array(filename, as_torch=True):
    """
        Returns an CODON msa MSA as a two dimension torch integer array
        (N, L) where N = Number of sequences in the MSA
                     L = Length of the protein (# of Amino Acid Residues)
               and each value in this array is the value of CODON_MAP
    """
    seq_iter = get_msa_from_aln_iter(filename)
    ret = np.array([codon_seq_to_int_list(seq) for seq in seq_iter], 
                        dtype=np.uint8) 
    if as_torch: 
        ret = torch.from_numpy(ret)
    return ret

def get_aa_msa_from_codon_msa(filename):
    """ 
        Reads a CODON MSA file and translates it to an integer array coded for
        Amino acids.
    """
    msa = get_codon_msa_as_int_array(filename)
    return CODON_AA_MAP[msa.long()]

def get_aa_msa_as_int_array(filename, as_torch=True):
    """
        Returns an AA msa MSA as a two dimension torch integer array
        (N, L) where N = Number of sequences in the MSA
                     L = Length of the protein (# of Amino Acid Residues)
               and each value in this array is the value of AA_MAP
    """
    seq_iter = get_msa_from_aln_iter(filename)
    aa_map = AA_MAP.copy()
    aa_map['-'] = 20 # add the gap character at the end
    ret = np.array([list(map(aa_map.get, seq)) for seq in seq_iter], 
                        dtype=np.uint8) 
    if as_torch: 
        ret = torch.from_numpy(ret)
    return ret


def string_to_prot(x):
    return np.array([AA_MAP[xi] for xi in x])
    
def prot_to_string(x):
    return AAs[x].tobytes().decode('ascii')

def calc_min_dist(x, m):
    """ x is a protein (in integer representation)
        m is another protein or an MSA"""
    m = m.squeeze()
    ret = None
    if m.ndim == 1:
        ret = (x != m).sum()
    elif m.ndim == 2:
        ret = (x != m).sum(axis=1).min()
    else:
        raise ValueError("m needs to have dimension 1 or 2")
    return ret


if __name__ == "__main__":
    msa_filename = MSA_FILESd[1]
    msa = get_codon_msa_as_int_array(msa_filename)
    print(msa.shape)


