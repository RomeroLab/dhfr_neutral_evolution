# About

This code repository accompanies the paper

**Inferring protein fitness landscapes from laboratory evolution experiments** \
Sameer D’Costa, Emily C. Hinds, Chase R. Freschlin,  Hyebin Song, Philip A. Romero

* [Biorxiv link](https://www.biorxiv.org/content/10.1101/2022.09.01.506224v1) [doi](https://doi.org/10.1101/2022.09.01.506224)

Data for this repository can be downloaded from NCBI Sequence Read Archive 
[Accession: PRJNA923701](https://www.ncbi.nlm.nih.gov/bioproject/923701)


## Downloading this repository
Use `git` to clone this repository from [github.com](https://github.com/RomeroLab/dhfr_neutral_evolution)


## Downloading raw sequencing data (Optional)

Raw sequencing data can be downloaded using the [convenience script](preprocessing/download_data.sh) in the `preprocessing` directory. 

```shell
(cd preprocessing; ./download_data.sh)
```

## Repository Installation

Use conda to install dependencies. The [main dependencies](environment.yml) are 

* `pytorch` version 1.6
* `python` version 3.8

There are several other dependencies like `numpy`, `matplotlib` etc that are
not fixed and the latest versions should be fine. 

```bash
# create conda `dhfr_analysis` environment
make install_conda

conda activate dhfr_analysis

# (Recommended) Adding python kernel in this conda environment to jupyter
#   notebook list of kernels
python3 -m ipykernel install --user

# make c++ extensions to python code
make ext
```

The exact the packages that this repository was tested on is saved at
[environment.explicit.txt](environment.explicit.txt)



### Running test script

```bash
# activate conda environment (if not already active)
conda activate dhfr_analysis

#run a test script 
# (This will print out a list of allowed arguments)
./runmodel.sh -h

```

The output should be 
```
usage: model.py [-h] [-l LEARNING_RATE] [-j] [-n NUM_EPOCHS] [-m MODEL_NAME]
                [-v LOG_LEVEL] [-o OUTPUT_DIRECTORY] [-t NT_TRANS_MAT_CSV]
                [-L PROTEIN_LENGTH] [--lam_main LAM_MAIN] [--lam_int LAM_INT]
                [-i] [--margin_penalty MARGIN_PENALTY]

optional arguments:
  -h, --help            show this help message and exit
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for ADAM optimizer
  -j, --disable_joblib  Disable saving of joblib cache
  -n NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        Number of training steps
  -m MODEL_NAME, --model_name MODEL_NAME
                        Model Name
  -v LOG_LEVEL, --log_level LOG_LEVEL
                        Log level for debugging messages
  -o OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Output directory for model params and figs (Path
                        Relative to the scripts directory)
  -t NT_TRANS_MAT_CSV, --nt_trans_mat_csv NT_TRANS_MAT_CSV
                        Nucleotide Transition Matrix pickle file (Path
                        Relative to the scripts directory)
  -L PROTEIN_LENGTH, --protein_length PROTEIN_LENGTH
                        Change protein length for debugging
  --lam_main LAM_MAIN   Regularization for main effects parameters
  --lam_int LAM_INT     Regularization for interaction effects parameters
  -i, --include_main_params
                        Include Main Parameters
  --margin_penalty MARGIN_PENALTY
                        Penalty for margin loss


```


### Reinstallation

These are the steps to reinstall the latest version of the code for the second
time after any changes.
```
# Make sure you are in the top directory of this repository

# pull the latest version of the code
git checkout master
git pull

# This deletes the `dhfr_analysis` conda environment without asking.  
make reinstall_conda

# clean up extra files
make deepclean
make ext
```


