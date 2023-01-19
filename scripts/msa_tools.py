"""Manipulate MSAs (stats, extract parts etc)

    Stripped down version of MSA tools copied from the VAEs repository. 
    Does not do reweighting or depend on torch or BioPython

  Typical usage example: (from the command line)
	#To get stats printed out to the terminal
        > python msa_tools.py \
                -i ../sequence_sets/cmx_aligned_blank_90.fasta \
                -w ../sequence_sets/cmx_wildtype.fasta  \
                -s ''
        Sampling 100k pairs to calculate pairwise distance
        avg_dist_from_wt_pct: 62.36%
        avg_pairwise_dist_pct: 63.12%
        name: cmx_aligned_blank_90
        num_seqs: 14441
        optimal_theta_after_mean_removal: 0.37
        seq_length: 559
        Time elapsed: 0.11 min
"""

import sys
import itertools
import functools # For CachedProperty
import contextlib # ExitStack to close filehandles
import gzip

import pathlib
import numpy as np

import yaml

def get_msa_from_aln(aln_filename, size_limit=None, 
                            as_numpy=True):
    """Reads a (plain text) aln file (can be gzipped also) and returns an MSA

    Takes a simple text file (ALN) which has one sequence per line. Returns 
    and MSA as a numpy array or as a list of Bio.Seq sequences.

    Args:
        aln_filename    : Filename or filename of ALN file to read
        size_limit      : Return upto size_limit sequences
        as_numpy        : return numpy byte array instead of list of seqs

    Returns:
        if as_numpy is False:
            A list of sequences in Bio.Seq format
        if as_numpy is True:
            A numpy byte array of dtpye S1 which represents the MSA. 
            The first axis is the sequence number. The second axis is the
            residue number. 
    """
    opener = open
    if aln_filename.endswith(".gz"):
        opener = gzip.open
    with opener(aln_filename, "rt") as fh:
        seq_io_gen = (line.strip() for line in fh)
        # Read only size_limit elements of the generator
        # if size_limit is None then we will read in everything
        seq_io_gen_slice = itertools.islice(seq_io_gen, size_limit) 
        seqs = (seq.upper() for seq in seq_io_gen_slice)
        ret = None
        if as_numpy:
            # convert to lists of lists for easy numpy conversion to 2D array
            ret = np.array([list(s) for s in seqs], dtype="|S1")
        else:
            ret = [Seq.Seq(s) for s in seqs]
        return ret


class MSA:
    """ Perform manipulations on MSA 

        Args:
            msa    : Numpy array of shape (N, L)  dtype "|S1"

    """

    def __init__(self, msa, name=""):
        self.msa = msa
        self.name = name

    @property
    def num_seqs(self):
        return self.msa.shape[0]

    @property
    def seq_length(self):
        return self.msa.shape[1]

    def calc_avg_dist_from_seq(self, seq):
        return (self.msa != seq).sum(axis=1).mean()


class MSAwt(MSA):
    """ Perform manipulations on MSA 

    Store an MSA as a numpy "|S1" array (most efficient form)
        TODO: Allow torch arrays later

    """

    def __init__(self, msa, wt, name=""):
        """
        Args:
            msa    : Numpy array of shape (N, L)  dtype "|S1"
            wt     : Numpy array of shape (1, L) or (L,)  dtype "|S1"
            name   : A string name for the MSA
        """
        self.wt = wt
        wt = wt.squeeze()
        if len(wt.shape) != 1:
            raise ValueError("WildType filename must have only one value in it")
        if wt.shape[0] != msa.shape[1]:
            raise ValueError("WildType and MSA files have different length sequences in them")
        super(MSAwt, self).__init__(msa=msa, name=name)

    @property
    def avg_dist_from_wt(self):
        return self.calc_avg_dist_from_seq(self.wt)

    @property
    def avg_dist_from_wt_pct(self):
        return self.avg_dist_from_wt  / self.seq_length * 100

    @property
    @functools.lru_cache(1) # save last calculation. Can use @cachedproperty for python 3.8+
    def avg_pairwise_dist(self):
        """ This is not a property because it takes forever to calculate """
        if self.num_seqs > 1000:
            print("Sampling 100k pairs to calculate pairwise distance")
            # sample pairwise indices
            it = (np.random.choice(self.num_seqs, 2, replace=True)
                    for _ in range(100000))
            distances = np.array([(self.msa[x, :] != self.msa[y, :]).sum() 
                                    for x, y in it])
        else:
            distances = np.array([self.calc_avg_dist_from_seq(self.msa[i, :]) 
                                    for i in range(self.seq_length)])
        return distances.mean()

    @property
    def avg_pairwise_dist_pct(self):
        return self.avg_pairwise_dist / self.seq_length * 100

    @property
    def optimal_theta_after_mean_removal(self):
        return 1 - (self.avg_dist_from_wt_pct + self.avg_pairwise_dist_pct) \
                        / (2 * 100)

    @property
    def optimal_plmc_theta_after_mean_removal(self):
        return 1 - self.optimal_theta_after_mean_removal

    def get_stats_dict(self):
        return {
            'name': self.name,
            'num_seqs': self.num_seqs,
            'seq_length': self.seq_length,
            'avg_dist_from_wt_pct': f"{self.avg_dist_from_wt_pct:.2f}%",
            'avg_pairwise_dist_pct': f"{self.avg_pairwise_dist_pct:.2f}%",
            'optimal_theta_after_mean_removal': \
                    float(round(self.optimal_theta_after_mean_removal, 2)),
            'optimal_plmc_theta_after_mean_removal': \
                    float(round(self.optimal_plmc_theta_after_mean_removal, 2))
            }

    def write_stats_dict(self, filetype):
        """ Write stats dictionary out to a file

            Args:
                filetype : str or filehandle (even something like sys.stdout)
        """
        with contextlib.ExitStack() as stack: # make sure any file handles are closed if opened
            fh = sys.stdout
            if isinstance(filetype, str):
                if filetype:
                    # Let's open a filehandle
                    p = pathlib.Path(filetype)
                    fh = stack.enter_context(p.open('wt'))
            else:
                fh = filetype # try printing directly to filetype
                # no need to close fh now as whatever passed it in will take care of it
            yaml.dump(self.get_stats_dict(), fh)

    def remove_seqs_below_mean(self, return_kept_indices=True):
        """ Return a new MSA with seqs below mean distance from WT removed 
            Args:
                return_kept_indices: If True a tuple (newMSA, kept_indices) 
                                        is returned
        """
        # make a shallow copy
        indices_to_keep = ~ self.get_indices_seqs_below_mean()
        ret = MSAwt(msa=self.msa[indices_to_keep, :],
                    wt = self.wt,
                    name = self.name)
        if return_kept_indices:
            ret = (ret, indices_to_keep)
        return (ret)

    def get_indices_seqs_below_mean(self):
        """ Get indices of sequences below avg distance from WT"""
        dist_from_wt = (self.msa != self.wt).sum(axis=1)
        return dist_from_wt < self.avg_dist_from_wt
        
    def write_msa_to_file(self, filename):
        """ Only supported writing gzip aln files for now"""
        filename = pathlib.Path(filename).with_suffix(".gz")
        with gzip.open(filename, "wt") as fh:
            for m in self.msa: 
                fh.write(m.tostring().decode("ascii")) 
                fh.write("\n") 


    @staticmethod
    def create_from_files(msa_filename, wildtype_filename):
        msa = get_msa_from_aln(msa_filename, as_numpy=True)
        wt = get_msa_from_aln(wildtype_filename, as_numpy=True)

        path = pathlib.Path(msa_filename)
        if path.suffix == ".gz":
            path = pathlib.Path(path.stem) # strip it off
        name = path.stem
        return MSAwt(msa, wt, name)


if __name__ == "__main__":
    import time
    import argparse
 
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--msa_filename",
                    help="input msa in ALN or FASTA format",
                    required=True) 
    parser.add_argument("-w", "--wildtype_filename",
                    help="filename that contains only WildType",
                    required=True) # required for now! TODO: make optional later
    parser.add_argument("-s", "--stats_filename", 
                    default=None, help="filename to save stats")
    parser.add_argument("-r", "--remove_below_mean_filename", 
                    help="filename to save new MSA with seqeuences " 
                         "below mean removed")
                         
    args = parser.parse_args()

    start_time = time.time()

    msa = MSAwt.create_from_files(args.msa_filename, args.wildtype_filename)

    if args.stats_filename is not None: # can be an empty string
        msa.write_stats_dict(args.stats_filename)

    if args.remove_below_mean_filename: # remove seqs closer than avg dist to WT
        msa_above_mean = msa.remove_seqs_below_mean(return_kept_indices=False)
        msa_above_mean.write_msa_to_file(args.remove_below_mean_filename)


    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))


