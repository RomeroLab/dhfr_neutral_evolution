#!/bin/bash

# environment to run VAE script
# echo "conda activate pytorch"

VAE_DIR=~/VAEs

python - <<'____HERE'
import numpy as np
x = np.linspace(-6., 6., 40)
z = np.vstack(list(map(lambda x: x.flatten(), np.meshgrid(x,x)))).T
np.save("./sample_latent_points.npy", z)
____HERE


awk -F, 'FNR >1 {print $1}' ../../working/adaptive_walk/even/walk_*.csv  \
        > adaptive_walks_trace_even_seqs.txt


# create latent space
cd $VAE_DIR/source
python vae_sample.py  \
        -i ~/neutral_evolution_data/DHFR/plot/sample_latent_points.npy \
        -o ~/neutral_evolution_data/DHFR/plot/sample_latent_points_seqs.txt \
        -l 186 \
        ../dhfrplot.yaml 

# redo their latent positions
python vae_latent_space.py  \
        -i ~/neutral_evolution_data/DHFR/plot/sample_latent_points_seqs.txt \
        -o ~/neutral_evolution_data/DHFR/plot/sample_latent_points_redo_latent.pkl \
        ../dhfrplot.yaml 

python vae_latent_space.py  \
        -i ~/neutral_evolution_data/DHFR/plot/adaptive_walks_trace_even_seqs.txt \
        -o ~/neutral_evolution_data/DHFR/plot/adaptive_walks_trace_even_seqs.pkl \
        ../dhfrplot.yaml 


