#!/bin/bash
cd /Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/FM2STRESS/FM2STRESS_project/code/../code/SKHASH2/SKHASH

# Activate conda pythoon environment named 'obspy' [********change miniconda3 to anaconda3 if using Anaconda]
source ~/miniconda3/etc/profile.d/conda.sh # or source ~/anaconda3/etc/profile.d/conda.sh in case of Anaconda
# conda init
conda activate obspy

python3 SKHASH.py /Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/FM2STRESS/FM2STRESS_project/code/../results/foc_mech/NCEDC_comp/HASH_IN_OUT/controlfile_NCEDC_PN_noQC_2008.txt
