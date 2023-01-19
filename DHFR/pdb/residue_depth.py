import Bio
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.PDBParser import PDBParser



if __name__ == "__main__":
    parser = PDBParser()
    structure = parser.get_structure("3K47", "3K47.pdb")
    model = structure[0]
    rd = ResidueDepth(model)

    with open("3K47_residue_depth.txt", "w") as fh_out:
        print("pdb_idx, uniprot_idx, pdb_res, avg_depth, ca_depth", file=fh_out)
        for res, dists in rd.property_list:
            print(f"{res.id[1]}, {res.id[1]+1}, {res.resname}, {dists[0]}, {dists[1]}", 
                    file=fh_out)
