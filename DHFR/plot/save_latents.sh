#!/bin/bash

# environment to run VAE script
# echo "conda activate pytorch"

VAE_DIR=~/VAEs


# get filenames of all rounds of sequencing data into array
msa_files=( `cd ../../scripts; python - <<'__HERE'
import config
for k,v in config.MSA_FILESd.items():
  print(k, v.absolute())
__HERE
`
)

# print out all the filenames of all the rounds we have data for
# We can create a latent space for each round 
# This is UNUSED for now
for element in "${msa_files[@]}"
do
   echo "${element}"
done


# copy natural sequences latent space
cp "${VAE_DIR}"/working/mDHFR_clean_latent.pkl .


# create a text file with sequences only to calculate latent space
DESIGNS_CSV_FILE=../../working/DHFR_incl_main_kit_taq_designed.csv
DESIGNS_OUTPUT_STUB=$(basename "${DESIGNS_CSV_FILE##*/}" .csv)

DESIGNS_TXT_FILE="${DESIGNS_OUTPUT_STUB}.txt"
DESIGNS_LATENT_FILE="${DESIGNS_OUTPUT_STUB}_latent.pkl"

# drop the header and the last sequence 
# the last sequence is the same as the second last sequence
# and signifies that the design process ended
cut -f1 -d, ${DESIGNS_CSV_FILE}  | sed '1d;$d' > "${DESIGNS_TXT_FILE}"

## CREATE LATENT SPACES

# create gen15 latent space
cd $VAE_DIR/source
python vae_latent_space.py  \
        -i ~/neutral_evolution_data/DHFR/Gen15/Gen15_aa.aln.gz \
        -o ~/neutral_evolution_data/DHFR/plot/Gen15_latent.pkl \
        ../dhfrplot.yaml 

# create designs latent space
python vae_latent_space.py  \
        -i ~/neutral_evolution_data/DHFR/plot/"${DESIGNS_TXT_FILE}" \
        -o ~/neutral_evolution_data/DHFR/plot/"${DESIGNS_LATENT_FILE}" \
        ../dhfrplot.yaml 

# shortest path
awk -F, 'NR > 1 {print $2}' \
        ../../working/DHFR_incl_main_kit_taq_mse_shortest_path.csv \
        > shortest_path.txt
python vae_latent_space.py \
        -i ~/neutral_evolution_data/DHFR/plot/shortest_path.txt \
        -o ~/neutral_evolution_data/DHFR/plot/shortest_path.pkl \
        ../dhfrplot.yaml 
