#!/bin/bash

# Script to create an MSA from uniprot database
# 1. Run jackhmmer to produce a stockholm alignment
# 2. Convert stockholm alignment to aligned fasta (afa)
# 3. Read this via python (skbio) and reduce the positions to match columns
# 4. Convert this fasta file to plain text

set -x

PROT="mDHFR"

# create fasta file
echo ">${PROT}" > ${PROT}.fasta ; cat ../mDHFR_WT.aln >> ${PROT}.fasta

jackhmmer \
    -A ${PROT}.sto \
    -o ${PROT}.out.txt \
	${PROT}.fasta /mnt/scratch/databases/uniprot/uniref90.fasta

# convert to aligned fasta format (afa)
/home/romeroroot/code/hmmer-3.1b2-linux-intel-x86_64/binaries/esl-reformat \
        -u -o ${PROT}.afa afa ${PROT}.sto 

echo "conda activate dhfr_analysis\nrun above line if required" 

python process.py

gzip ${PROT}_clean.fasta

echo rm ${PROT}.out.txt ${PROT}.sto ${PROT}.afa
