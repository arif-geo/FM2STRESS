#!/bin/bash

# Change directory into PhaseNet
cd /Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/FM2STRESS/FM2STRESS_project/code/../../../../Software_docs/PhaseNet

# Activate conda pythoon environment named 'phasenet' [********change miniconda3 to anaconda3 if using Anaconda]
source ~/miniconda3/etc/profile.d/conda.sh # or source ~/anaconda3/etc/profile.d/conda.sh in case of Anaconda
# conda init
conda activate phasenet

# Run Phasenet prediction
python phasenet/predict.py \
  --model=model/190703-214543 \
  --data_list=/Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/FM2STRESS/FM2STRESS_project/code/../results/phasenet_diting_others/phasenet_files/mseed_list.csv \
  --data_dir=/Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/FM2STRESS/FM2STRESS_project/code/../data/eq_data/2_waveforms \
  --format=mseed \
  --amplitude \
  --batch_size=1 \
  --sampling_rate=100 \
  --result_dir=/Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/FM2STRESS/FM2STRESS_project/code/../results/phasenet_diting_others/phasenet_files \
  --result_fname=phasenet_phasepick_3d_grid_new\
  --min_s_prob=0.95 \

# Deactivate virtual environment (optional)
# conda deactivate

