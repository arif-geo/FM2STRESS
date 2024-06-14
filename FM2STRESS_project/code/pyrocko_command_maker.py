#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyperclip

year = 2008
ncedc_pick_dir = '/Users/mdaislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Github/FM2STRESS/FM2STRESS_project/data/NCEDC_picks'
research_dir_local = '/Users/mdaislam/research_arif/MTJ_waveform_data_Arif/Data_NCEDC_events'

marker_file = f'{ncedc_pick_dir}/{year}_NCEDC_picks_above_slab_pyrocko_PN.txt'
mseed_dir = f'{research_dir_local}/mseed/{year}'
station_xml = f'{research_dir_local}/stationxml/00_{year}_inventory.xml'

cmd = f"snuffler {mseed_dir}/*.mseed --stationxml={station_xml} --markers={marker_file}" #--event=00_master_event_file_PyRocko.txt
pyperclip.copy(cmd)
print(f"go to mseed folder.\nCommand is copied to clipboard:\n {cmd}")