#!/bin/bash

# Change directory into PhaseNet
cd /Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/earth_sci_data_analysis_arif/Final_project/code/../../../../Software_docs/PhaseNet/

# Activate conda pythoon environment named 'phasenet'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phasenet

# Run Phasenet prediction
python phasenet/predict.py \
  --model=model/190703-214543 \
  --data_list=/Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/earth_sci_data_analysis_arif/Final_project/code/../data/eq_data/all_data/phasenet_files/mseed_list.csv \
  --data_dir=/Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/earth_sci_data_analysis_arif/Final_project/code/../data/eq_data/all_data \
  --format=mseed \
  --amplitude \
  --batch_size=1 \
  --sampling_rate=100 \
  --result_dir=/Users/mdarifulislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/earth_sci_data_analysis_arif/Final_project/code/../data/eq_data/all_data/phasenet_files \
  --result_fname=phasenet_phasepick_datetime \
  --add_polarity \
  # --plot_figure

# Deactivate virtual environment (optional)
# conda deactivate

