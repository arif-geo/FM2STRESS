# write the bin/bash python
#!/bin/bash python 

import os
import sys
from tqdm.auto import tqdm
import glob
import importlib

import pandas as pd
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read_inventory, Inventory, read, Stream
import matplotlib.pyplot as plt

# my custom functions
sys.path.append("./my_funcs")
from get_waveforms_parallel_v3 import get_waveforms_parallel, get_station_inventory

# set up the project directory
project_dir = "../"
data_dir = os.path.join(project_dir, "data/eq_data")
catalog_dir = os.path.join(data_dir, "1_eq_catalogs")
waveform_dir = os.path.join(data_dir, "2_waveforms")
station_dir = os.path.join(data_dir, "3_station_catalogs")

# get a list of mseed files without folder names
mseed_files = [os.path.basename(f) for f in glob.glob(f"{waveform_dir}/*.mseed")]

# client_list = ['IRIS', 'NCEDC', 'SCEDC']
# channels_string = 'HH*,BH*,HN*,EH*'                   # by default given to get_waveforms_parallel function
# priority_channels = ['HH*', 'BH*', 'HN*', 'EH*']

progress_bar = tqdm(total=len(selected_eq), desc="Downloading events", dynamic_ncols=True)
for i, row in selected_eq.iterrows():
    event_id = row.id
    # check if the event data is already downloaded
    if f"{event_id}.mseed" in mseed_files:
        progress_bar.update(1)
        continue

    event_time = UTCDateTime(pd.to_datetime(row.time))
    starttime = event_time - 30
    endtime = event_time + 120

    xml_file = f"{waveform_dir}/xml/{event_id}_event_inv.xml"
    if os.path.exists(xml_file):
        inv = read_inventory(xml_file)

    else:
        inv = get_station_inventory(starttime, endtime) # client_list, priority_channels are default
        inv.write(xml_file, format="STATIONXML")
        inv = read_inventory(xml_file)
    
    print(f"Downloading waveforms for {event_id} at {event_time}")

    # get the waveforms for the event and save them
    st, inv = get_waveforms_parallel(starttime, endtime, inventory=inv) # client_list, priority_channels are default
    
    print(event_id, len(st))
    if len(st) > 0:
        st.write(f"{waveform_dir}/{event_id}.mseed", format="MSEED")
        # inv.write(f"{waveform_dir}/{event_id}_event_inv.txt", format="STATIONTXT")
        # remove the inventory file
        os.remove(f"{waveform_dir}/xml{event_id}_event_inv.xml")
        progress_bar.update(1)
        break

    else:
        progress_bar.update(1)
        # continue
        break  