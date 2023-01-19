import sys
import gzip
import numpy as np

WRAP = 70

def print_a2m_from_iter(seq_iter, fh=sys.stdout):
    for i, line in enumerate(seq_iter):
        print(">seqid_{}/1-186".format(i), file=fh)
        line = line.rstrip()
        N = len(line)
        for wc in range(int(N / WRAP)):
            print(line[wc*WRAP:(wc+1)*WRAP], file=fh)
        print(line[((wc+1)*WRAP):], file=fh)

if __name__ == "__main__":
    import contextlib
    filename = sys.argv[1]
    with contextlib.ExitStack() as cm:
        if len(sys.argv) > 2:
            fh_out = cm.enter_context(open(sys.argv[2], "wt"))
        else:
            fh_out = sys.stdout
        it = ()
        if filename[-3:] == ".gz":
            gzfile = cm.enter_context(gzip.open(filename, "rt"))
            it =  (line.rstrip() for line in gzfile)
        elif filename[-4:] == ".pkl":
            np_load = np.load(filename, allow_pickle=True)[1]
            it = (seq[1:-1] for seq in np_load)
        elif filename[-4:] == ".aln":
            fh = cm.enter_context(gzip.open(filename, "rt"))
            it = (line.rstrip() for line in fh)
        else:
            raise ValueError("Cannot understand file extension")
        print_a2m_from_iter(it, fh = fh_out)
