#!/bin/bash
cd /Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/FM2STRESS/FM2STRESS_project/code/../code/SKHASH/SKHASH

# Activate conda pythoon environment named 'phasenet' [********change miniconda3 to anaconda3 if using Anaconda]
source ~/miniconda3/etc/profile.d/conda.sh # or source ~/anaconda3/etc/profile.d/conda.sh in case of Anaconda
# conda init
conda activate obspy

python3 SKHASH.py examples/maacama_SKHASH_MTJ/control_test.txt
