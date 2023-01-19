""" Extract distance_matrix or contacts from a pdb file """

import numpy as np
import logging

import Bio
import Bio.PDB


# definition from 
# https://github.com/sanderlab/3Dseq/blob/master/AAC6%20scripts/pdbdists3.m
backbone_atoms = {'C','N','O','OXT'};  # not side-chain atoms
backbone_atoms_non_gly = {*backbone_atoms, "CA"}  

def extract_chain(filename, model_num, chain_id):
    """Gets a chain from a pdb filename """
    parser = Bio.PDB.PDBParser()
    structure = parser.get_structure("generic", args.pdb_filename)
    logging.info("Loaded structure from file : %s", args.pdb_filename)

    logging.info("Number of models: %s", len(structure))
    model = structure[args.model_num]
    logging.info("Selected model : %s", args.model_num)

    logging.info("Number of chains in model %s: %s", args.model_num, len(model))
    chain = model[args.chain_id]
    logging.info("Selected chain : %s", args.chain_id)

    return chain

def extract_cb_atoms(residue):
    """ Take a residue and return a list of atoms that we need to compute
        distances. In this case it is all CB atoms. For Gly we pick the CA atom
    """ 
    residue_id = residue.get_id()
    ret = None
    if residue_id[0] == " ": # it isn't a hetero-residue or water
        atom_type = "CB"
        if residue.get_resname() == "GLY":
            atom_type = "CA"
        ret = [residue[atom_type]]  # single atom is picked, list returned
    return ret

def extract_sidechain_atoms(residue):
    """ Take a residue and return a list of atoms that we need to compute
        distances. In this case it is all the side chain atoms. For Gly we 
        include the CA atom. For all other Amino acids we elimiate it.
    """ 
    ret = None
    residue_id = residue.get_id()
    if residue_id[0] == " ": # it isn't a hetero-residue or water
        ba = backbone_atoms_non_gly
        if residue.get_resname() == "GLY": # add back CA atom
            ba = backbone_atoms
        ret = [a for a in residue if (a.get_id() not in ba)]
    return ret

def compute_mean_coordinate(atom_list):
    return np.vstack([a.get_coord() for a in atom_list]).mean(axis=0)


def coord_dist_calc(loc1, loc2):
    diff = loc1 - loc2
    return np.sqrt(np.dot(diff, diff))


def sidechain_dist_calc(atom_list1, atom_list2):
    mean1 = compute_mean_coordinate(atom_list1)
    mean2 = compute_mean_coordinate(atom_list2)
    return coord_dist_calc(mean1, mean2)


def compute_distance_mat(chain, extract_atoms_func, compute_distance_func):
    """ Take a chain and extract atoms and then compute distances using
        extracted atoms 
        FIXME: Later. Calculate the distance between two chains. Then this
        functionality can be done by passing in the same chain twice.
    """
    resatoms = [] # list of residues containing list of atoms 
    for residue in chain:
        atoms = extract_atoms_func(residue)
        if atoms is not None:
            resatoms.append(atoms)
    L = len(resatoms)
    logging.info("Number of residues picked: %d", L)

    dist_mat = np.zeros((L,L), dtype=np.float)

    for res_seq_id1, al1 in enumerate(resatoms):
        for res_seq_id2, al2 in enumerate(resatoms):
            if res_seq_id2 > res_seq_id1:
                continue
            else:
                dist_mat[res_seq_id1, res_seq_id2] = \
                        compute_distance_func(al1, al2)
                dist_mat[res_seq_id2, res_seq_id1] = \
                        dist_mat[res_seq_id1, res_seq_id2]

    return dist_mat

def cb_distance(chain):
    """ Compute the distance between CB atoms """
    return compute_distance_mat(chain, 
            extract_atoms_func=extract_cb_atoms,
            compute_distance_func=lambda x,y : x[0] - y[0])

def sidechain_distance(chain):
    """ Compute distance between centers of sidechain atoms """
    return compute_distance_mat(chain,
            extract_atoms_func=extract_sidechain_atoms,
            compute_distance_func=sidechain_dist_calc)


def get_5_8_contact_format(dist_mat, min_residue_sep):
    """ Mark "contacts" on a matrix of distances
        distances less than 8A are marked 8
        distances less than 5A are marked 5
        All other distances are marked as 0
        if residue_sequence_seperation is less than min_residue_sep
            then it overrides all distances and is marked as 0
    """
    contact_map = np.zeros(dist_mat.shape, dtype=np.int)
    contact_map[dist_mat < 8] = 8
    contact_map[dist_mat < 5] = 5 # the 5s will overwrite the 8s
    c_x, c_y = np.indices(contact_map.shape)
    close_residues_mask = (np.abs(c_x - c_y) < min_residue_sep)
    contact_map[close_residues_mask] = 0

    return contact_map

def count_contacts_in_contact_map(contact_map):
    return (contact_map > 0).sum() // 2


if __name__ == "__main__":
    import time
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--pdb_filename",
                        help="input PDB file", required=True)
    parser.add_argument("-m", "--model_num",
                        help="model number (0-start indexing) ",
                        default=0, type=int, required=False)
    parser.add_argument("-c", "--chain_id",
                        help="Chain ID", default="A", required=False)
    parser.add_argument("-l", "--log_level",
                        help="Logging level", default="INFO", required=False)
    parser.add_argument("-d", "--distance_matrix_filename",
                        help="Output filename for distance matrix in npy fmt",
                        default=None, required=False)
    parser.add_argument("--dist_calc",
                        help="Function to use to extract atoms "
                             "and compute distances",
                        default="cb_distance", required=False)
    parser.add_argument("--contact_5_8_filename",
                        help="Contact map in distance 5 and distance 8 format",
                        default=None, required=False)
    parser.add_argument("--contact_min_sep",
                        help="Minimum separation for contact. default >= 5",
                        default=5, required=False)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    p = pathlib.Path(args.pdb_filename)
    chain = extract_chain(filename = p, model_num = args.model_num,
                            chain_id = args.chain_id)

    distance_calculator = locals()[args.dist_calc]
    logging.info("Setting distance calculator to : %s", args.dist_calc)

    dist_mat = distance_calculator(chain)

    dist_mat_filename = args.distance_matrix_filename
    if dist_mat_filename is None:
        dist_mat_filename = p.parent / (p.stem + "_distances.npy")
    np.save(dist_mat_filename, dist_mat)

    contact_5_8_filename = args.contact_5_8_filename
    if contact_5_8_filename is None:
        contact_5_8_filename = p.parent / (p.stem + "_5_8_contacts.npy")
    contact_5_8 = get_5_8_contact_format(dist_mat, 
                        min_residue_sep = args.contact_min_sep)
    logging.info("Number of contacts in contact_mat : %d", 
                    count_contacts_in_contact_map(contact_5_8))
    np.save(contact_5_8_filename, contact_5_8)

