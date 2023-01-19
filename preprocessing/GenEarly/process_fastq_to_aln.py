""" Script to process raw fastq files

    Take a fasta.gz that is already processed and save reads that are the 
    right length into an alignment file. Also we remove sequences with stops in
    the middle of the sequence.
"""

import gzip
import numpy as np
import skbio

# Reference sequence for merging
# taken from Emily's notes
# https://github.com/RomeroLab/sameerd/blob/master/projects/DHFR_Neutral_Evolution/doc/Emily/filepaths_and_notes.md
refseq = "CATATGGTTCGACCATTGAATTGTATTGTAGCAGTATCACAAAATATGGGTATTGGTAAAAATGG" \
         "TGATTTGCCATGGCCACCTCTACGTAATGAATTCAAGTATTTTCAGCGAATGACTACTACTAGTT" \
         "CAGTTGAAGGTAAACAAAATTTAGTTATTATGGGTCGTAAAACATGGTTTAGTATCCCTGAAAAA" \
         "AATCGACCATTAAAAGATCGAATCAATATTGTTTTAAGTCGTGAATTAAAAGAACCTCCACGAGG" \
         "TGCTCATTTTTTAGCTAAAAGTTTAGATGATGCATTGCGACTAATCGAACAACCAGAATTGGCAT" \
         "CAAAAGTTGATATGGTATGGATTGTAGGTGGTAGTTCAGTTTATCAGGAAGCAATGAATCAACCT" \
         "GGTCACTTACGATTGTTTGTTACTCGAATCATGCAGGAATTTGAAAGTGATACTTTTTTTCCAGA" \
         "AATTGATTTGGGTAAATATAAATTACTACCTGAATATCCAGGTGTTCTAAGTGAAGTTCAGGAAG" \
         "AAAAAGGTATCAAATATAAATTTGAAGTTTATGAGAAGAAAGATTGAGCT"
N = len(refseq) # should be 570


if __name__ == "__main__":
    import time
    import argparse
    import pathlib
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_filename",
                    help="input fastq file gzipped (extension .fastq.gz)", 
                    required=True)
    parser.add_argument("-o", "--output_filename_nts",
                    help="output nucleotide sequence file gzipped (extension .aln.gz)")
    parser.add_argument("-a", "--output_filename_aa",
                    default=os.devnull,
                    help="output Amino Acid sequence file gzipped (extension .aln.gz)")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_filename)

    output_path_nts = None
    output_path_aa = None
    if args.output_filename_nts is None:
        # put the output in the same place as the input
        input_real_stem = input_path.stem.split('.')[0] 
        output_path_nts = input_path.parent / (input_real_stem + "_nts")
        output_path_aa = input_path.parent / (input_real_stem + "_aa")
        output_path_nts = output_path_nts.with_suffix(".aln.gz")
        output_path_aa = output_path_aa.with_suffix(".aln.gz")
    else:
        output_path_nts = pathlib.Path(args.output_filename_nts)
        output_path_aa = pathlib.Path(args.output_filename_aa)

    print("Input       : ", input_path)
    print("Output (NTS): ", output_path_nts)
    print("Output (AA) : ", output_path_aa)
    print("N           : ", N)

    start_time = time.time()
    with gzip.open(input_path, "rt") as fh_inp, \
        gzip.open(output_path_nts, "wt") as fh_out_nts, \
        gzip.open(output_path_aa, "wt") as fh_out_aa:
        for seq in skbio.read(fh_inp, format="fastq", variant="illumina1.8"): 
            if len(seq) == N:
                seq_str = seq.values.tobytes().decode('ascii')
                # write out sequence without start and stop codons
                seq_str = seq_str[6:(N-6)]
                aa_seq = skbio.DNA(seq_str).translate()
                if not aa_seq.has_stops():
                    print(seq_str, file=fh_out_nts)
                    aa_seq_str = aa_seq.values.tobytes().decode('ascii')
                    print(aa_seq_str, file=fh_out_aa)

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

