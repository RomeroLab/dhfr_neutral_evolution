import functools
import operator

import numpy as np
import Bio
import Bio.AlignIO
import Bio.Data
import Bio.SeqIO

prot = "mDHFR"
prot_length = 186
length_cutoff = 0.50 # drop sequences with less than this fraction of prot

print(f"Reading alignment file")
msa = Bio.AlignIO.read("mDHFR.afa", format="fasta")
print(f"Number of sequences in alignment file: {len(msa)}")

last_seq = msa[-1]
# last line should be the fasta sequence we did the query with
assert last_seq.name == prot

match_idx = [i for i,a in enumerate( last_seq.seq ) if a != "-"]
assert len(match_idx) == prot_length 

## Smaller msa with only match columns
print("FILTER: Filtering columns to match query sequence") 
small_msa = functools.reduce(operator.add, (msa[:, i:(i+1)] for i in match_idx))
print(f"Number of sequences : {len(small_msa)}")

last_seq = small_msa[-1]
# last line should be the fasta sequence we did the query with
assert last_seq.name == prot


print("FILTER: Filtering sequences on length") 
# filter resulting proteins on their length
length_filter_msa = Bio.Align.MultipleSeqAlignment([
        r for r in small_msa if len(r.seq.ungap()) >= length_cutoff*prot_length ])
print(f"Number of sequences : {len(length_filter_msa)}")

invalid_map = str.maketrans('', '', Bio.Data.IUPACData.protein_letters + '-')
print("FILTER: Filtering sequences with invalid sequences") 
clean_msa = Bio.Align.MultipleSeqAlignment([
        r for r in length_filter_msa if 
        not len(str(r.seq).translate(invalid_map))]) 
print(f"Number of sequences : {len(clean_msa)}")

Bio.SeqIO.write(clean_msa, f"{prot}_clean.fasta", "fasta")

