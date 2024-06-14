import os
import glob
import numpy as np
import pandas as pd
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader
from obspy import UTCDateTime, read, read_inventory, Inventory, Stream

dl_path = '/Users/mdaislam/research_arif/MTJ_waveform_data_Arif/Data_NCEDC_events'
preffered_channels = ['HHZ', 'BHZ', 'HNZ', 'EHZ', 'SHZ', 'DPZ', 'SLZ', 'CNZ', 'HN3', 'HH3', 'ENZ']
ncedc_eqcat_df = pd.read_csv(f'../data/NCEDC_picks/NCEDC_eq_cat_above_slab.csv', parse_dates=['time'])
print(ncedc_eqcat_df.head(5))
# group by year
ncedc_eqcat_df['year'] = pd.to_datetime(ncedc_eqcat_df['time']).dt.year

for iyear, year_cat_df in ncedc_eqcat_df.groupby('year'):
    for i, row in year_cat_df[:].iterrows():
        if os.path.exists(f'{dl_path}/mseed/{iyear}/{row.id}.mseed'):
            continue
        origin_time = UTCDateTime(row.time)
        domain = CircularDomain(
            latitude=row.latitude, longitude=row.longitude, 
            minradius=0.0, maxradius=1.5)   
        
        restrictions = Restrictions(
            starttime=origin_time - 30,
            endtime=origin_time + 120,
            reject_channels_with_gaps=True,
            minimum_length=0.95,
            channel_priorities=preffered_channels)
        mdl = MassDownloader(providers=["NCEDC", "IRIS"])
        mdl.download(
            domain, restrictions,
            mseed_storage=f'{dl_path}/temp/waveforms',
            stationxml_storage=f'{dl_path}/temp/stationxmls')

        # Join all mseed and cleanup
        st = Stream()
        for file in glob.glob(f'{dl_path}/temp/waveforms/*.mseed'):
            st += read(file)
        st.resample(100)

        # Clean up the inventory
        inv = Inventory()
        for xml in glob.glob(f'{dl_path}/temp/stationxmls/*.xml'):
            inv += read_inventory(xml)
        temp_inv = f'{dl_path}/temp/temp_inv.txt'
        inv.write(temp_inv, 'STATIONTXT')
        inv = pd.read_csv(temp_inv, delimiter='|').drop_duplicates(subset=['#Network', 'Station', 'Location', 'Channel'])
        inv.to_csv(temp_inv, sep='|', header=True, index=False)
        inv = read_inventory(temp_inv, format='STATIONTXT')
        
        # Remove temp files in temp folder
        os.system(f'rm -r {dl_path}/temp')
  
        os.makedirs(f'{dl_path}/mseed/{iyear}', exist_ok=True)
        os.makedirs(f'{dl_path}/stationxml/{iyear}', exist_ok=True)
        
        # Write to file
        st.write(f'{dl_path}/mseed/{iyear}/{row.id}.mseed', format='MSEED')
        inv.write(f'{dl_path}/stationxml/{iyear}/{row.id}.xml', format='STATIONXML')

    #break
    print(f'Year {iyear} done.\nTotal events: {len(year_cat_df)}\n')
