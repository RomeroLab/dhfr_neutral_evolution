#!/bin/bash

#SRA project page
# https://www.ncbi.nlm.nih.gov/bioproject/923701

# Biosample page
# https://www.ncbi.nlm.nih.gov/biosample?LinkName=bioproject_biosample_all&from_uid=923701

set -x

EARLY_RAW_DATA_DIR=GenEarly/raw_data
mkdir -p "${EARLY_RAW_DATA_DIR}"


IDENTIFIERS=( SRR23078416 SRR23078415 SRR23078414 SRR23078413 SRR23078412 )

for ID in ${IDENTIFIERS[@]}
do
  fastq-dump --outdir "${EARLY_RAW_DATA_DIR}" --split-files -A "${ID}"
done

GEN15_RAW_DATA_DIR=Gen15/raw_data
mkdir -p "${EARLY_RAW_DATA_DIR}"
fastq-dump --outdir "${GEN15_RAW_DATA_DIR}" -A SRR23078411 

