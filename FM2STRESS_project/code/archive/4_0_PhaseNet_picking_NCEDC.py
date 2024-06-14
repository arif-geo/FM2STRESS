#!/Users/mdaislam/anaconda3/envs/phasenet

# Phase Picking (PhaseNet)

import os
import glob
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read_inventory, Inventory, read, Stream
from classes_functions.other_fun import make_phasenet_script

# Function to list files in a directory
def my_list_files(directory, file_extension=None):
    import glob
    file_path_list = glob.glob(f'{directory}/*{file_extension}')
    file_list = [os.path.basename(f) for f in file_path_list]
    return file_list

data_dir = '/Users/mdaislam/research_arif/MTJ_waveform_data_Arif/Data_NCEDC_events'
phasenet_git_dir = '/Users/mdaislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Software_docs/PhaseNet'
out_dir = os.path.abspath(
    os.path.join(
        os.getcwd(),'../results/phasenet_diting_others/phasenet_picks_NCEDC_cat'
        ))

# 
years_dirs = my_list_files(f'{data_dir}/mseed', file_extension='')
for year in years_dirs:
    print(f'Processing year: {year}')
    # Make a list of .mseed files into a csv
    mseed_files = my_list_files(f'{data_dir}/mseed/{year}', '.mseed')
    pd.DataFrame(mseed_files, columns=['fname']
        ).to_csv(f'{out_dir}/{year}_mseed_list.csv', index=False, header=True)
    
    # make bash script for running PhaseNet
    script_content = make_phasenet_script(
        phasenet_git_dir,
        waveform_dir=f'{data_dir}/mseed/{year}',
        data_list=f'{out_dir}/{year}_mseed_list.csv',
        result_dir=f'{out_dir}/{year}',
        result_fname=f'{year}_phasenet_picks.csv',
        conda_env='anaconda3',
    )

    # Write the script content to a file
    with open(f"{out_dir}/run_phasenet.sh", "w") as f:
        f.write(script_content)
    
    # Make the script executable
    print("Making the script executable...")
    os.system(f"chmod +x {out_dir}/run_phasenet.sh")
    
    # Run the script
    run = input("Do you want to run PhaseNet now? (y/n): ")
    if run.lower() == "y":
        print("Running PhaseNet...")
        os.system(f"{out_dir}/run_phasenet.sh")
    else:
        print("PhaseNet already run. Skipping...")
    
    break

#