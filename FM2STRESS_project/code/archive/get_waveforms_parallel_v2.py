import os
import time
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read_inventory, Inventory, read, Stream
from tqdm.notebook import tqdm
from multiprocessing import Pool
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("download_log.txt")
logger.addHandler(file_handler)

def download_station(args):
    client, networks, starttime, endtime, output_folder, priority_channels = args
    """
    Download waveforms and station information for a specific station.
    """
    for i, network in enumerate(networks):
        print(f"network: {network}")
        # get stations in the current network
        stations = inventory.networks[i].stations
        print(f"stations: {len(stations)}")
        for station in stations:
            if station.code in success_stn:
                logger.info(f"{station.code} already downloaded. Skipping...")
                continue

            channels = station.channels
            channels_list = [channel.code[0:2] for channel in channels]
            channels_list = list(set(channels_list))

            for priority_channel in priority_channels:
                if priority_channel[0:2] not in channels_list:
                    continue

                try:
                    temp_st = client.get_waveforms(
                        network=network,
                        station=station.code,
                        location="*",
                        channel=priority_channel,
                        starttime=starttime,
                        endtime=endtime,
                    )
                    logger.info(f"Waveform downloaded: {network}.{station.code}.{priority_channel} from {client.name}.")
                    # st += temp_st

                    # Download station information
                    try:
                        inv = client.get_stations(
                            network=network,
                            station=station.code,
                            location="*",
                            channel=priority_channel,
                            starttime=starttime,
                            endtime=endtime,
                            level="response",
                        )
                        # inv.write(f"{output_folder}/{network}.{station.code}_{client.name}.xml", format="STATIONXML")
                        # inv.write(f"{output_folder}/{network}.{station.code}_{client.name}.txt", format="STATIONTXT")
                        
                        st += temp_st
                        success_stn.append(station.code)
                        logger.info(f"Inventory downloaded: {len(success_stn)} out of {len(inventory.get_contents()['stations'])} stations.")
                        
                        # since we have found the right channel, we can break the loop
                        break

                    except Exception as e:
                        logger.error(f"Failed to download station information for {network}.{station.code}.{priority_channel} from {client.name}: {e}")
                        fid = open(f"{output_folder}/failed_download.txt", 'a')
                        fid.close()
                        continue

                except Exception as e:
                    logger.error(f"Failed to download waveforms for {network}.{station.code}.{priority_channel} from {client.name}: {e}")
                    continue

def get_waveforms_parallel(client_list, inventory, event_id, starttime, endtime, output_folder, priority_channels):
    """
    Get waveforms for a specific event from a list of clients with parallel processing.

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

    global success_stn
    global st
    success_stn = []
    st = Stream()

    logger.info(f"Downloading waveforms for event {event_id}...")

    # Parallelize the loop over stations
    args_list = []
    for client_name in client_list:
        client = Client(client_name, debug=False, timeout=30)
        logger.info(f"Fetching data from {client_name}...")
        networks = inventory.get_contents()['networks']
        

    # Parallelize the loop over stations
    pool = Pool(processes=4)
    with tqdm(total=len(args_list)) as pbar:
        for _ in pool.imap_unordered(download_station, args_list):
            pbar.update(1)

    # Wait for all tasks to finish
    pool.close()
    pool.join()

    logger.info(f"Download completed. Total successful stations: {len(success_stn)} out of {len(inventory.get_contents()['stations'])}.")

    # Write waveforms to file
    # st.write(f"{output_folder}/{event_id}.mseed", format="MSEED")

