#!/bin/bash

## All relative paths should be relative to the scripts directory
## It would be better to pass in absolute paths to this script
## if any paths are used

cd scripts
# compile any extensions that need compiling
#make all 

python3 model.py "$@"
 
