zcat ../mDHFR_clean.fasta.gz  > mDHFR_clean.fasta

PLMC_BIN=~/software/plmc/bin/plmc

${PLMC_BIN} -o mDHFR.model_params -c mDHFR.couplings.txt -f mDHFR \
        -m 150 -lh 1.0 -le 37.0 -t 0.2 -g mDHFR_clean.fasta   > mDHFR.plmc.txt 2>&2


# conda activate evcouplings
PYTHONPATH=~/software/EVmutation:$PYTHONPATH python3 mut_matrix.py
