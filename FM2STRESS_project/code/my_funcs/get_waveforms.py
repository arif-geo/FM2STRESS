import os
import time
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read_inventory, Inventory, read, Stream
from tqdm.notebook import tqdm

def get_waveforms(client_list, inventory, event_id, starttime, endtime, output_folder, priority_channels):
    """
    Get waveforms for a specific event from a list of clients.

    Args:
        client_list (list): List of client names.
        event_id (str): Event ID.
        starttime (str): Start time of the event in UTC format.
        endtime (str): End time of the event in UTC format.
        output_folder (str): Output folder for the waveforms.
        priority_channels (list): List of priority channels to download.

    Returns:
        None
    """

    success_stn = []
    st = Stream()

    for i, client_name in tqdm(enumerate(client_list)):
        client = Client(client_name, debug=False, timeout=60)
        print(f"{'*'*50}\nFetching data from {client_list[i]}... \n {'*'*50}")
        networks = inventory.get_contents()['networks']

        for i, network in tqdm(enumerate(networks)):
            # get stations in the current network
            stations = inventory.networks[i].stations

            for station_content in tqdm(stations):
                station = station_content.code

                if station in success_stn:
                    print(f"{station} already downloaded. Skipping...")
                    continue
                else: # station not in success_stn:
                    channels = station_content.channels
                    channels_list = [channel.code[0:2] for channel in channels]
                    channels_list = list(set(channels_list)) # remove duplicates
                    # print(f"channels_list: {network}.{station}: {channels_list}")
                    # continue

                    # match the right channels
                    for priority_channel in priority_channels:
                        if priority_channel[0:2] not in channels_list:
                            continue
                        else:
                            channel = priority_channel

                            # download the waveform
                            try:
                                print(f"{network}.{station}.{channel}.{starttime}.{endtime}")
                                temp_st = client.get_waveforms(
                                    network=network,
                                    station=station,
                                    location="*",
                                    channel=channel,
                                    starttime=starttime,
                                    endtime=endtime,
                                )
                                # st += temp_st
                                print(f"{'*'*8} waveform downloaded {network}.{station}.{channel} from {client_name}.")

                                # download station information
                                try:
                                    print("getting STATION lavel inventory...")
                                    inv = client.get_stations(
                                        network=network,
                                        station=station,
                                        # location="*",
                                        # channel=channel, 
                                        starttime=starttime,
                                        endtime=endtime,
                                        level="station" #"response",
                                    )
                                    
                                    st += temp_st
                                    inv.write(f"{output_folder}/{network}.{station}_{client_name}.xml", format="STATIONXML")
                                    inv.write(f"{output_folder}/{network}.{station}_{client_name}.txt", format="STATIONTXT")
                                    
                                    success_stn.append(station)
                                    print(f" inventory downloaded: {len(success_stn)} out of {len(inventory.get_contents()['stations'])} stations")
                                    
                                    # since we have found the right channel, we can break the loop
                                    break

                                except:
                                    # print(f"        Failed to download station information for {network}.{station}.{channel} from {client_name}.")
                                    fid = open(f"{output_folder}/failed_download.txt", 'a')
                                    fid.close()
                                    continue
                            except:
                                # print(f"Failed to download waveforms for {network}.{station}.{channel} from {client_name}.")
                                continue
    # write waveforms to file
    # st.write(f"{output_folder}/{event_id}.mseed", format="MSEED")
    print(f"Total success: {len(success_stn)} out of {len(inventory.get_contents()['stations'])} stations")