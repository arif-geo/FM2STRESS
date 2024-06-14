#!/bin/bash

# Change directory into PhaseNet
cd /Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Software_docs/PhaseNet

# Activate conda pythoon environment named 'phasenet' [********change miniconda3 to anaconda3 if using Anaconda]
source ~/minconda3/etc/profile.d/conda.sh 
# conda init
conda activate phasenet

# Run Phasenet prediction
python phasenet/predict.py \
  --model=model/190703-214543 \
  --data_list=/Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/FM2STRESS/FM2STRESS_project/results/phasenet_diting_others/phasenet_picks_NCEDC_cat/2008_mseed_list.csv \
  --data_dir=/Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Data/NCEDC_events_data/mseed/2008 \
  --format=mseed \
  --amplitude \
  --batch_size=1 \
  --sampling_rate=100 \
  --result_dir=/Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/FM2STRESS/FM2STRESS_project/results/phasenet_diting_others/phasenet_picks_NCEDC_cat/2008 \
  --result_fname=2008_phasenet_picks\
  --min_p_prob=0.7 \
  --min_s_prob=1.0 \

# Deactivate virtual environment (optional)
# conda deactivate obspy

# --add_polarity \
# --plot_figure
  