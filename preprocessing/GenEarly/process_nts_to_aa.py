""" Script to translate nts to aa files

    Take a nts.aln.gz that is already processed and translate to aa.aln.gz
"""

import gzip
import numpy as np
import skbio


if __name__ == "__main__":
    import time
    import argparse
    import pathlib
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_filename_nts",
                    help="output nucleotide sequence file gzipped (extension .aln.gz)")
    parser.add_argument("-a", "--output_filename_aa",
                    help="output Amino Acid sequence file gzipped (extension .aln.gz)")
    args = parser.parse_args()

    input_path_nts = pathlib.Path(args.input_filename_nts)
    output_path_aa = pathlib.Path(args.output_filename_aa)

    print("Input (NTS) : ", input_path_nts)
    print("Output (AA) : ", output_path_aa)

    start_time = time.time()
    with gzip.open(input_path_nts, "rt") as fh_inp_nts, \
        gzip.open(output_path_aa, "wt") as fh_out_aa:
            for seq in fh_inp_nts:
                seq_str = seq.strip()
                aa_seq = skbio.DNA(seq_str).translate()
                aa_seq_str = aa_seq.values.tobytes().decode('ascii')
                print(aa_seq_str, file=fh_out_aa)

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

