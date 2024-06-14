import numpy as np
import pandas as pd
from geopy.distance import geodesic


# -------------------- Add event to station distance column to phasenet polarity file --------------------
def add_sta_dist_to_polpick(eq_catalog_file, phasenet_pick_file, station_inventory_file):

    # read eq catalog file
    eq_df = pd.read_csv(eq_catalog_file)

    # read phasenet polarity file
    phasenet_df = pd.read_csv(phasenet_pick_file)
    
    # read station inventory file
    sta_df = pd.read_csv(station_inventory_file, 
                        sep='|', 
                        header=0,
                        usecols=[0, 1, 2, 4, 5]) # Station|Latitude|Longitude  
    # keep only unique station names
    sta_df = sta_df.drop_duplicates(subset=['Station'])
    # Make  a new Network.Station column
    sta_df['station_id'] = sta_df['#Network'] + '.' + sta_df['Station']

    # group the phasenet picks by 'file_name' aka event_id
    # loop through each event_id, find the event's latitude and longitude from eq_df
    # find the station's latitude and longitude from sta_df
    # calculate the distance between the event and the station
    # add the distance to the phasenet_df

    # create a new column in phasenet_df to store the distances
    phasenet_df['sta_dist_km'] = 1000.0 # initialize with a large value

    stoper = 0
    for evfilename, PNgroup in phasenet_df.groupby('file_name'):
        event_id = evfilename.split('.')[0]

        # get the event's latitude and longitude
        event_row = eq_df.loc[eq_df.id==event_id]
        elat, elon, edep = event_row.latitude.values[0], event_row.longitude.values[0], event_row.depth.values[0]
        
        for i, row in PNgroup.iterrows():
            station_id = row.station_id.split('.')[0] + '.' + row.station_id.split('.')[1] #net.sta
            # get the station's latitude and longitude
            if station_id not in sta_df.station_id.values:
                print(f"Station {station_id} ")
                continue
            slat = sta_df.loc[sta_df.station_id==station_id, 'Latitude'].values[0]
            slon = sta_df.loc[sta_df.station_id==station_id, 'Longitude'].values[0]

            # calculate the distance between the event and the station
            dist = np.round(geodesic((elat, elon), (slat, slon)).km, 2)
            # print(f"{i}:station_id: {station_id}, distance: {dist} km")

            # add the distance to the phasenet_df
            phasenet_df.loc[i, 'sta_dist_km'] = dist
            # i remains the same as for the original phasenet_df


    return phasenet_df # with the new column 'sta_dist' added



# -------------------- PhaseNet run.sh file generator --------------------  

def make_phasenet_script(
  phasenet_git_path, 
  waveform_dir,
  data_list=None,
  result_dir=None,
  result_fname='phasenet_phasepick_TEST',
  min_p_prob=0.70,
  conda_env='miniconda3',
  ):

  """# Make a script to run PhaseNet 
  # *** VVI: every directory is with respect to the location of this notebook ***
  input:
      phasenet_git_path: path to the source code of PhaseNet 
  """
  
  script_content = f"""#!/bin/bash

# Change directory into PhaseNet
cd {phasenet_git_path}

# Activate conda pythoon environment named 'phasenet' [********change miniconda3 to anaconda3 if using Anaconda]
source ~/{conda_env}/etc/profile.d/conda.sh 
conda init
conda activate phasenet

# Run Phasenet prediction
python phasenet/predict.py \\
  --model=model/190703-214543 \\
  --data_list={data_list} \\
  --data_dir={waveform_dir} \\
  --format=mseed \\
  --amplitude \\
  --batch_size=1 \\
  --sampling_rate=100 \\
  --result_dir={result_dir} \\
  --result_fname={result_fname}\\
  --min_p_prob={min_p_prob} \\
  --min_s_prob=1.0 \\

# Deactivate virtual environment (optional)
# conda deactivate obspy

# --add_polarity \\
# --plot_figure
  """
  
  return script_content
