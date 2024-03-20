import os
import time
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read_inventory, Inventory, read, Stream
from tqdm import tqdm
from multiprocessing import Pool

def get_waveforms_parallel(client_list, inventory, event_id, starttime, endtime, output_folder, priority_channels):
    """
    Get waveforms for a specific event from a list of clients.
    This function will use parallel processing to download waveforms from multiple clients at the same time.

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

    def download_station(client, network, stations):
        """
        Download waveforms and station information for a specific station.
        input:
            client_name: name of the client (str)
            network: network code (str)
            station: station object (obspy.station; A type of xml data)
        return:
            None
            downloads waveforms and station information to the output folder
        """
        for station_content in stations:
            station = station_content.code

            if station in success_stn:
                print(f"{station} already downloaded. Skipping...")
                continue
            else: # station_content.code not in success_stn:
                channels = station_content.channels
                channels_list = [channel.code[0:2] for channel in channels]
                channels_list = list(set(channels_list)) # remove duplicates

                # match the right channels
                for priority_channel in priority_channels:
                    if priority_channel[0:2] not in channels_list:
                        continue
                    else:
                        channel = priority_channel

                        # download the waveform
                        try:
                            temp_st = client.get_waveforms(
                                network=network,
                                station=station,
                                location="*",
                                channel=channel,
                                starttime=starttime,
                                endtime=endtime,
                            )
                            # st += temp_st
                            print(f"{'*'*8} Waveform downloaded {network}.{station}.{channel} from {client_name}.")

                            # download station information
                            try:
                                inv = client.get_stations(
                                    network=network,
                                    station=station.code,
                                    location="*",
                                    channel=channel, 
                                    starttime=starttime,
                                    endtime=endtime,
                                    level="response",
                                )
                                
                                st += temp_st #append the waveform to the stream

                                # inv.write(f"{output_folder}/{network}.{station}_{client_name}.xml", format="STATIONXML")
                                # inv.write(f"{output_folder}/{network}.{station}_{client_name}.txt", format="STATIONTXT")
                                
                                success_stn.append(station.code)
                                print(f" Inventory downloaded: {len(success_stn)} out of {len(inventory.get_contents()['stations'])} stations")
                                
                                # since we have found the right channel, we can break the loop
                                break

                            except:
                                # print(f"        Failed to download station information for {network}.{station.code}.{channel} from {client_name}.")
                                fid = open(f"{output_folder}/failed_download.txt", 'a')
                                fid.close()
                                continue
                        except:
                            # print(f"Failed to download waveforms for {network}.{station.code}.{channel} from {client_name}.")
                            continue
    # ==================== end of download_station function ====================                     

    # Parallelize the loop over stations
    pool = Pool(processes=4)
#===
    for i, client_name in enumerate(client_list):
        client = Client(client_name, debug=False, timeout=30)
        print(f"{'*'*50}\nFetching data from {client_list[i]}... \n {'*'*50}")
        networks = inventory.get_contents()['networks']

        for i, network in enumerate(networks):
            # get stations in the current network
            stations = inventory.networks[i].stations
            pool.apply_async(download_station, args=(client, network, stations))
            # Wait for all tasks to finish
    pool.close()
    pool.join()
#===
    
    # write waveforms to file
    # st.write(f"{output_folder}/{event_id}.mseed", format="MSEED")
    print(f"Total success: {len(success_stn)} out of {len(inventory.get_contents()['stations'])} stations")
    # st.plot();