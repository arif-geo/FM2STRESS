#!/Users/mdarifulislam/miniconda3/envs/obspy/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import obspy
from obspy import read, UTCDateTime
import os
import pyperclip

from classes_functions.my_class_funcs import MyFileConverter, SkhashRunner
mfc = MyFileConverter()
# %reload_ext autoreload
# %autoreload 2

# Set up directories
project_dir = f".."
skhash_dir = f"{project_dir}/code/SKHASH2/SKHASH"  ## SKHASH2 is the most recent version
skhash_mtj_dir = f'{skhash_dir}/examples/maacama_SKHASH_MTJ'

# in and out directories [WRT SKHASH.py]
in_dir = f'{skhash_mtj_dir}/one_by_one/IN'
out_dir = f'{skhash_mtj_dir}/one_by_one/OUT'
os.makedirs(in_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

# Set up input files
pyrocko_marker_file = f'{project_dir}/results/phasenet_diting_others/' + '00_master_marker_ASSOCIATED_v3.txt'
eq_cat_file = f'{project_dir}/data/eq_data/1_eq_catalogs/' + 'usgs_catalog_2008-01-01_2024-01-01_M2.csv'
inv_file = f'{project_dir}/data/eq_data/3_station_catalogs/' + '00_station_inventory_master.txt'

# step 1: convert pyrocko picks to skhash format polarity csv file
# read pyrocko marker file into a dictionary
pyrocko_dict = mfc.PyrockoMarker2Dict(pyrocko_marker_file)

# empty dataframe to store pyrocko picks in phasenet-like format
pyrocko_pnlike_df = pd.DataFrame(
    columns=['station_id', 'phase_time', 'phase_score', 'phase_type', 'file_name', 'diting_polarity','diting_sharpness','pyrocko_phase_time','pyrocko_polarity']
    )

# Select an event
inp = int(input('0-47:'))
key = list(pyrocko_dict.keys())[inp]
# for key in pyrocko_dict.keys():
evarray = np.array(pyrocko_dict[key])
df = pd.DataFrame({
    'station_id': evarray[:,0],
    'pyrocko_phase_time': evarray[:,1],
    'phase_type': evarray[:,2],
    'pyrocko_polarity': evarray[:,3],
    'file_name': key
})
pyrocko_pnlike_df = pd.concat([pyrocko_pnlike_df, df], axis=0) 
# print(pyrocko_pnlike_df)

pyrocko_pnlike_df = pyrocko_pnlike_df.replace('None', np.nan).dropna(subset=['pyrocko_polarity'])
# print(pyrocko_pnlike_df)

# convert pyrocko marker to skhash format
skrun = SkhashRunner()

# polarity file [SKHASH format]
skhash_pol_pyroko = skrun.PhaseNet2SKHASH_polarity(
    eq_cat_file, 
    PN_picks_path=pyrocko_pnlike_df,
    pyrocko_only=True,
    )

# step 2: make station file [SKHASH format]
if not os.path.exists(f'{in_dir}/station_all.csv'):
    skhash_station_df = skrun.make_SKHASH_station_file(given_inventory=inv_file,
        keep_Z_only = True, drop_duplicates = True, output_path = None,
        )
    skhash_station_df.to_csv(f'{in_dir}/station_all.csv', index=False)

# step 3: Make control file [SKHASH format]
in_dir1 = f'examples/maacama_SKHASH_MTJ/one_by_one/IN'
out_dir1 = f'examples/maacama_SKHASH_MTJ/one_by_one/OUT'
control_file = skrun.edit_skhash_control_file(
    conpfile = f'{in_dir1}/{key}_pol_concensus.csv',
    stfile = f'{in_dir1}/station_all.csv',
    vmodel_paths = 'examples/velocity_models_MTJ/vz.MTJ.txt',
    outfile1 = f'{out_dir1}/{key}_out.csv',
    outfile2 = f'{out_dir1}/{key}_out2.csv',
    outfile_pol_agree = f'{out_dir1}/{key}_out_polagree.csv',
    outfile_pol_info = f'{out_dir1}/{key}_out_polinfo.csv',
    outfolder_plots = f'{out_dir1}/plots',
    plot_station_names = True,
    plot_acceptable_solutions = True,
    delmax = input('Enter max stn-ev distance in km [1-200 | 200 km default]: ') or 200,
    )
# print(control_file)

# write file to input dir
skhash_pol_pyroko.to_csv(f'{in_dir}/{key}_pol_concensus.csv', index=False)
with open(f'{in_dir}/0_oneByOne_controlfile.txt', 'w') as f:
    f.write(control_file)

# step 4: Run SKHASH
# cd into SKHASH directory first
cmd = f'python3 SKHASH.py examples/maacama_SKHASH_MTJ/one_by_one/IN/0_oneByOne_controlfile.txt'
print('Go to SKHASH directory and run the following command: \n')
print(cmd, '\n')
pyperclip.copy(cmd)

# step Misc: add the event to done_events
if not os.path.exists(f'{in_dir}/done_events.txt'):
    done_events = np.array([inp, key]).reshape(1,2)
    np.savetxt(f'{in_dir}/done_events.txt', done_events, fmt='%s')
else:
    done_events = np.loadtxt(f'{in_dir}/done_events.txt', dtype=str)
    if key not in done_events:
        done_events = np.vstack([done_events, [inp, key]])
        np.savetxt(f'{in_dir}/done_events.txt', done_events, fmt='%s')


# if focal mechanism is done, make station distribution plot
    try:
        if input('Done running SKHASH? (y/n):') == 'y':
            import classes_functions.plot_FM as plot_FM
            event_id = key
            # read the station polarity file and get the event location
            event_sta_pol_df, elat, elon, edep = plot_FM.get_sta_lat_lon_pol(
                event_id, 
                skhash_pol_file=f'{in_dir}/{event_id}_pol_concensus.csv', 
                sta_inv_file=inv_file
            )

            # get the focal mechanism parameters
            strike, dip, rake, fqual, _ = plot_FM.get_FM_params(
                skhash_out_file=f'{out_dir}/{event_id}_out.csv',
                event_id=event_id
            )

            # make the plot
            fig = plot_FM.plot_FM_stations(
                event_id, event_sta_pol_df, 
                elat, elon, edep,
                strike, dip, rake,
                region=[-126, -122., 39., 42]
            )
            fig.basemap(frame=[f"+t{event_id} - Quality:{fqual}"])
            fig.savefig(f'{out_dir}/plots/{event_id}_station_distribution.png')
            fig.show()
    except:
        pass

# Now run PyRocko to check the picks
# get absolute filepaths
stationxml = f'{project_dir}/data/eq_data/3_station_catalogs/' + '00_station_inventory_master.xml'
stationxml = os.path.abspath(stationxml)
marker_file = os.path.abspath(pyrocko_marker_file)
cmd = f"snuffler ../processed/{key}.mseed --stationxml={stationxml} --markers={marker_file}"
pyperclip.copy(cmd)
print("Station distribution plot is saved. \n Iside mseed folder run the following command: \n")
print(cmd)
print('Last run event:', inp, key)