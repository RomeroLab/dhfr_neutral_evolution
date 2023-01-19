""" Script to process raw fastq files

    This is run after the paired-end reads have been merged with the FLASH
    program
"""

import gzip
import skbio


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
    parser.add_argument("-q", "--quality_cutoff",
                    help="Keep sequence if all Q(base) >= quality_cutoff", 
                        type=int, required=True)
    parser.add_argument("-L", "--min_length",
                    help="Keep sequences >= min_length", type=int, required=True)
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_filename)
    quality_cutoff = args.quality_cutoff
    min_length = args.min_length

    output_path = None
    if args.output_filename is None:
        # put the output in the same place as the input
        input_real_stem = input_path.stem.split('.')[0] + \
                            f"_Q{quality_cutoff}"
        output_path = input_path.parent / input_real_stem
        output_path = output_path.with_suffix(".fastq.gz")
    else:
        output_path = pathlib.Path(args.output_filename)

    print("Input :    ", input_path)
    print("Output:    ", output_path)
    print("QCutoff:   ", quality_cutoff)
    print("MinLength: ", min_length)

    start_time = time.time()
    with gzip.open(input_path, "rt") as fh_inp, \
        gzip.open(output_path, "wt") as fh_out:
        for seq in skbio.read(fh_inp, format="fastq", variant="illumina1.8"): 
            if len(seq) < min_length:
                continue
            seq_qual = seq.positional_metadata['quality']
            if (seq_qual >= quality_cutoff).all():
                seq.write(fh_out, format="fastq", variant="illumina1.8")
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

