#!/bin/bash

# Directory where the raw reads are stored
RAW_DATA_DIR=raw_data

#IDENTIFIERS=( L1_A2_S1 L2_A4_S2 L4_A12_S4 L5_A16_S5 L6_A19_S6 )
#R1_EXT=_L001_R1_001.fastq.gz
#R2_EXT=_L001_R2_001.fastq.gz

IDENTIFIERS=( SRR23078416 SRR23078415 SRR23078414 SRR23078413 SRR23078412 )
R1_EXT=_1.fastq
R2_EXT=_2.fastq

WORKING_DATA_DIR=working

QUALITY_CUTOFF=15
COMPOUND_CUTOFF=10
MIN_LENGTH=500

if [ ! -d "$RAW_DATA_DIR" ]; then
    echo "$RAW_DATA_DIR does not exist."
    exit 1
fi

mkdir -p ${WORKING_DATA_DIR}

ROUND_NUM=1
for ID in ${IDENTIFIERS[@]}
do  
	R1_FILE="${RAW_DATA_DIR}/${ID}_L001_R1_001.fastq.gz"
	R2_FILE="${RAW_DATA_DIR}/${ID}_L001_R2_001.fastq.gz"
    # Process raw data with flash
    flash ${R1_FILE} ${R2_FILE} \
            -z \
            -d ${WORKING_DATA_DIR} \
            -o Round${ROUND_NUM}

    # Filter each read so that it satisfies constraints on quality and length
    STITCHED_FILE=${WORKING_DATA_DIR}/Round${ROUND_NUM}.extendedFrags.fastq.gz
    SIMPLE_FILE=${WORKING_DATA_DIR}/Round${ROUND_NUM}_Q${QUALITY_CUTOFF}.fastq.gz
    python process_fastq_simple.py  \
        -i ${STITCHED_FILE} \
        -o ${SIMPLE_FILE} \
        -q ${QUALITY_CUTOFF} \
        -L ${MIN_LENGTH}

    # Compound quality filter for entire read
    COMPOUND_FILE=${SIMPLE_FILE%%.*}_C${COMPOUND_CUTOFF}.fastq.gz
    python process_fastq_compound.py \
        -i ${SIMPLE_FILE} \
        -o ${COMPOUND_FILE} \
        -C ${COMPOUND_CUTOFF}
    
    NTS_OUTPUT_FILE=${COMPOUND_FILE%%.*}_nts.aln.gz
    python process_fastq_to_aln.py \
        -i ${COMPOUND_FILE} \
        -o ${NTS_OUTPUT_FILE}

    ((ROUND_NUM++))
done

