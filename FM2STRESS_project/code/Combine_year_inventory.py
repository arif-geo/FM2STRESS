#!~/anaconda3/envs/obspy

import os
import pandas as pd
from obspy import read_inventory, Inventory

cwd = os.getcwd()
xml_dir = f'{cwd}/research_arif/MTJ_waveform_data_Arif/Data_NCEDC_events/stationxml/'
year_dirs = sorted([f for f in os.listdir(xml_dir) if os.path.isdir(os.path.join(xml_dir, f))])

for year in year_dirs:
    print(f'Processing year: {year}')
    xml_files = [f for f in os.listdir(f'{xml_dir}/{year}') if f.endswith('.xml')]
    year_inv = Inventory()

    for xml_file in xml_files:
        inv = read_inventory(f'{xml_dir}/{year}/{xml_file}')
        year_inv += inv

    year_inv.write(f'{xml_dir}/.temp.txt', 'STATIONTXT')
    print('Before: ', len(year_inv.get_contents()['stations']))
    year_inv = pd.read_csv(
        f'{xml_dir}/.temp.txt', sep='|', header=0
        ).drop_duplicates(subset=['#Network', 'Station', 'Location', 'Channel']
        ).sort_values(by=['#Network', 'Station', 'Location', 'Channel']
        ).reset_index(drop=True)
    print('After: ', len(year_inv))
    year_inv.to_csv(f'{xml_dir}/.temp.txt', sep='|', index=False)
    year_inv = read_inventory(f'{xml_dir}/.temp.txt', format='STATIONTXT')
    year_inv.write(f'{xml_dir}/00_{year}_inventory.xml', 'STATIONXML')
    os.remove(f'{xml_dir}/.temp.txt')
    # break