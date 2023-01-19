#!/bin/bash

# environment to run VAE script
# echo "conda activate pytorch"

VAE_DIR=~/VAEs

ADAPTIVE_WALKS_DIR="../../working/adaptive_walk"

ADAPTIVE_WALKS_START_LIST="adaptive_walks_start_seqs.txt"
ADAPTIVE_WALKS_TRACE_LIST="adaptive_walks_trace_seqs.txt"

# Add Wildtype
echo "Adding wildtype sequence"
(cd ../../scripts ; python -c "import config; print(config.WT_AA)") \
        > "${ADAPTIVE_WALKS_START_LIST}"
echo "Adding starting sequence from each adaptive walk"
# For exach adaptive walk extract the start_protein
for json_file in "${ADAPTIVE_WALKS_DIR}"/walk_*.json;
do
    python extract_sequence_from_adaptive_walk_json.py \
            "${json_file}" >> "${ADAPTIVE_WALKS_START_LIST}"
done
echo "Adding convergent sequence from adaptive walks"
# add the sequence where all the adaptive walks converge # end_protein
python extract_sequence_from_adaptive_walk_json.py \
    "${ADAPTIVE_WALKS_DIR}/walk_1.json" -f end_protein \
    >> "${ADAPTIVE_WALKS_START_LIST}"

echo "Adding trace sequences from a few select adaptive walks"
# adding list of sequences as we trace out the trajectories of some walks 
echo -n > "${ADAPTIVE_WALKS_TRACE_LIST}"
for trace_file in "${ADAPTIVE_WALKS_DIR}"/walk_*.csv;
do
    awk -F, 'NR > 1 {print $1}' "${trace_file}" >> \
                "${ADAPTIVE_WALKS_TRACE_LIST}"
done


## copy natural sequences latent space
## this is already done by the ./save_latents.sh scripts
#cp "${VAE_DIR}"/working/mDHFR_clean_latent.pkl .

ADAPTIVE_WALKS_START_LATENT_FILE="adaptive_walks_start_latent.pkl"
ADAPTIVE_WALKS_TRACE_LATENT_FILE="adaptive_walks_trace_latent.pkl"

## CREATE LATENT SPACES

echo "Creating latent spaces"

# create adaptive walks start proteins latent space  + WT + convergent protein
cd $VAE_DIR/source
python vae_latent_space.py  \
        -i ~/neutral_evolution_data/DHFR/plot/"${ADAPTIVE_WALKS_START_LIST}" \
        -o ~/neutral_evolution_data/DHFR/plot/"${ADAPTIVE_WALKS_START_LATENT_FILE}" \
        ../dhfrplot.yaml 

# create adaptive walks tracing latent space
python vae_latent_space.py  \
        -i ~/neutral_evolution_data/DHFR/plot/"${ADAPTIVE_WALKS_TRACE_LIST}" \
        -o ~/neutral_evolution_data/DHFR/plot/"${ADAPTIVE_WALKS_TRACE_LATENT_FILE}" \
        ../dhfrplot.yaml 


