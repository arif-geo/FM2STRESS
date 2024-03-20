import os
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import read_inventory, UTCDateTime, Stream, Inventory
from tqdm import tqdm
from multiprocessing import Pool
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(f"../data/eq_data/download_log.txt")
logger.addHandler(file_handler)

def download_station(args):
    """
    Download waveforms and station information for a specific station.
    This function will be used in parallel processing.
    """
    client, network, station, priority_channels, starttime, endtime, output_folder, st, inv, success_stn = args
    st_local = Stream()
    inv_local = Inventory()

    for priority_channel in priority_channels:
        ch_list = list(set([ch.code[0:2] for ch in station.channels])) # list(set()) removes duplicates
        if priority_channel[0:2] not in ch_list:
            continue
        # print(f"{network.code}.{station.code}.[{priority_channel}]")

        # check if this station has already been downloaded for a priority channel
        # if #1 (HH*) is downloaded, skip #2 (BH*) etc.
        if station.code in success_stn:
            break
        try:
            # print(f'{network.code}.{station.code}.{priority_channel}_{starttime}_{endtime}')
            temp_st = client.get_waveforms(
                network=network.code,
                station=station.code,
                location="*",
                channel=priority_channel,
                starttime=starttime,
                endtime=endtime,
            )

            # Download station information
            try:
                temp_inv = client.get_stations(
                    network=network.code,
                    station=station.code,
                    location="*",
                    channel=priority_channel,
                    starttime=starttime,
                    endtime=endtime,
                    level="response",
                )

                # if station.code not in success_stn:
                success_stn.append(station.code)
                st_local += temp_st
                inv_local.networks.extend(temp_inv.networks)
                # print(f"successful stream and inv: {network.code}.{station.code}.{priority_channel}")
                
                
                logger.info(f"Waveform downloaded: {network.code}.{station.code}.{priority_channel} from {client.base_url}.")
                logger.info(f"Inventory downloaded: {len(success_stn)} out of {len(inventory.get_contents()['stations'])} stations.")
                
                # Break the loop if the right channel is found
                break

            except Exception as e:
                logger.error(f"Failed to download station information for {network.code}.{station.code}.{priority_channel} from {client.base_url}: {e}")

        except Exception as e:
            logger.error(f"Failed to download waveforms for {network.code}.{station.code} from {client.base_url}: {e}")
    return st_local, inv_local

#==================================================================================================

def get_waveforms_parallel(client_list, inventory, starttime, endtime, output_folder, priority_channels):
    """
    Get waveforms for a specific event from a list of clients with parallel processing.

    Args:
        client_list (list): List of client names.
        starttime (str): Start time of the event in UTC format.
        endtime (str): End time of the event in UTC format.
        output_folder (str): Output folder for the waveforms.
        priority_channels (list): List of priority channels to download.

    Returns:
        None
    """
    global success_stn
    success_stn = []
    waveforms_list = []
    inv_list = []
    st = Stream()
    inv = Inventory()

    logger.info(f"Downloading waveforms...")
    
    # Prepare arguments for download_station function
    args_list = []
    for client_name in client_list:
        client = Client(client_name, debug=False, timeout=30)

        for network in inventory.networks:
            for station in network.stations:
                args_list.append((client, network, station, priority_channels, starttime, endtime, output_folder, st, inv, success_stn))

    # Parallelize the loop over stations
    pool = Pool(processes=6)
    with tqdm(total=len(args_list), disable=True) as pbar:
        for temp_st, temp_inv in pool.imap_unordered(download_station, args_list):
            pbar.update(1) # update progress bar for each station
            st += temp_st
            inv.networks.extend(temp_inv.networks)

    # Wait for all tasks to finish
    pool.close()
    pool.join()

    # Write waveforms to file
    # st.write(f"{output_folder}/event_waveforms.mseed", format="MSEED")
    # # inv.write(f"{output_folder}/event_inventory.xml", format="STATIONXML")
    # inv.write(f"{output_folder}/{}event_inventory.txt", format="STATIONTXT")

    return st, inv