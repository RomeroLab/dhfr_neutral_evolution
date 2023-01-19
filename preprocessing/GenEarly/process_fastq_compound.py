""" Script to process raw fastq files

    This is run after the paired-end reads have been merged with the FLASH
    program
"""

import gzip
import numpy as np
import skbio


def calc_compound_score(seq_qual_values):
    a = 1 - np.power(10., -(seq_qual_values / 10))
    return -10 * np.log10(1 - np.cumprod(a)[-1])


if __name__ == "__main__":
    import time
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_filename",
                    help="input fastq file gzipped (extension .fastq.gz)", 
                    required=True)
    parser.add_argument("-o", "--output_filename",
                    help="output fastq file gzipped (extension .fastq.gz)")
    parser.add_argument("-C", "--compound_cutoff",
                    help="Keep sequence if Qcomp(seq) > compound_cutoff", 
                        type=float, required=True)
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_filename)
    compound_cutoff = args.compound_cutoff

    output_path = None
    if args.output_filename is None:
        # put the output in the same place as the input
        input_real_stem = input_path.stem.split('.')[0] + \
                f"_C{compound_cutoff:0}"
        output_path = input_path.parent / input_real_stem
        output_path = output_path.with_suffix(".fastq.gz")
    else:
        output_path = pathlib.Path(args.output_filename)

    print("Input :      ", input_path)
    print("Output:      ", output_path)
    print("QcompCutoff: ", compound_cutoff)

    start_time = time.time()
    with gzip.open(input_path, "rt") as fh_inp, \
        gzip.open(output_path, "wt") as fh_out:
        for seq in skbio.read(fh_inp, format="fastq", variant="illumina1.8"): 
            seq_qual = seq.positional_metadata['quality']
            seq_qual_comp_score = calc_compound_score(seq_qual.values)
            if seq_qual_comp_score > compound_cutoff:
                seq.write(fh_out, format="fastq", variant="illumina1.8")

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

