import pathlib
import gzip
import numpy as np

import skbio

WT_str = 'VRPLNCIVAVSQNMGIGKNGDLPWPPLRNEFKYFQRMTTTSSVEGKQNLVIMGRKTWFSIPEKN' \
         'RPLKDRINIVLSRELKEPPRGAHFLAKSLDDALRLIEQPELASKVDMVWIVGGSSVYQEAMNQP' \
	 'GHLRLFVTRIMQEFESDTFFPEIDLGKYKLLPEYPGVLSEVQEEKGIKYKFEVYEKKD'
WT = skbio.Protein(WT_str)

if __name__ == "__main__":
    for p in sorted(pathlib.Path(".").glob("Round*_Q15_C10_aa.aln.gz")):
        print(p)
        with gzip.open(p, 'rt') as fh:
            data = np.array([list(x.rstrip()) for x in fh], dtype="S1")
        N = data.shape[0]
        wt_sim = np.sum(data == WT.values, axis=1) / len(WT_str)
        wt_sim_mean = wt_sim.mean()
        print(f"\tNum sequences: {N}")
        print(f"\tAvg sim to WT: {wt_sim_mean*100:.2f}%")
