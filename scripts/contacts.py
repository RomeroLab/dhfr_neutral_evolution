""" Functions to help with evaluating contact prediction """

import numpy as np
import config

def with_read_plmc_coupling_scores_filename(filename, sort=True):
    """Reads in the coupling scores file of the plmc program and returns an
       array of scores in zero-based indexing. Optional argument to sort
    """
    with open(filename, "rt") as fh:
        return read_plmc_coupling_scores(fh, sort=sort)

def read_plmc_coupling_scores(fh, sort=True):
    """Reads in the coupling scores file of the plmc program and returns an
       array of scores in zero-based indexing. Optional argument to sort
    """
    coupling_scores = []
    for line in fh:
        res_i, focus_i, res_j, focus_j, zero, score = line.strip().split()
        coupling_scores.append((int(res_i)-1, int(res_j)-1, float(score)))
    scores = np.array(coupling_scores, dtype=
                [('idx1', np.int), ('idx2', np.int), ('score', np.float)])
    if sort: # sort inplace and in descending order
        scores[::-1].sort(order="score")
    return(scores)


def get_DHFR_contact_matrix():
    """ Load in the DHFR contact matrix 
        5 marks residue pairs <5A apart
        8 marks residue pairs 5-8A apart
        0 marks residue pairs >8A apart (no contact)
    """
    return np.load(config.CONTACT_MAP_MATRIX_FILENAME)

