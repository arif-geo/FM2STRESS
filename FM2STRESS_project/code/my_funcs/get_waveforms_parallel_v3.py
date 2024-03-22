import os
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import read_inventory, UTCDateTime, Stream, Inventory
from tqdm import tqdm
from multiprocessing import Pool
import logging


#==================================================================================================

def download_station(args):
    """
    Download waveforms and station information for a specific station.
    This function will be used in parallel processing.
    """
    client, network, station, priority_channels, starttime, endtime, st, inv, success_stn = args
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

            # if station.code not in success_stn:
            success_stn.append(station.code)
            st_local += temp_st
            inv_local.networks.extend(temp_inv.networks)

            # Break the loop if the right channel is found
            break

        except Exception as e:
            # print(f"failed: {network.code}.{station.code}.{priority_channel}")
            # continue to the next channel
            pass

    return st_local, inv_local

#==================================================================================================

def get_waveforms_parallel(
    starttime, endtime, 
    inventory = None, 
    client_list = ['IRIS', 'NCEDC'],#, 'SCEDC'],
    priority_channels = ['HH*', 'BH*', 'HN*', 'EH*'],
    output_folder = None,
    ):
    """
    Get waveforms for a specific event from a list of clients with parallel processing.

    Args:
        client_list (list): List of client names.
        starttime (str): Start time of the event in UTC format.
        endtime (str): End time of the event in UTC format.
        output_folder (str): Output folder for the waveforms.
        priority_channels (list): List of priority channels to download.

        inventory (obspy.Inventory): *** Must be `stationxml` format ***

    Returns:
        None
    """
    global success_stn
    success_stn = []
    waveforms_list = []
    st = Stream()
    inv = Inventory()
    
    # Prepare arguments for download_station function
    args_list = []
    for client_name in client_list:
        client = Client(client_name, debug=False, timeout=30)

        for network in inventory.networks:
            for station in network.stations:
                args_list.append((client, network, station, priority_channels, starttime, endtime, st, inv, success_stn))

    # Parallelize the loop over stations
    #get core count
    core_count = os.cpu_count() - 1 # leave one core for other processes
    pool = Pool(core_count) # use all available cores

    with tqdm(total=len(args_list), disable=True) as pbar:
        for temp_st, temp_inv in pool.imap_unordered(download_station, args_list):
            pbar.update(1) # update progress bar for each station
            st += temp_st
            inv.networks.extend(temp_inv.networks)

    # Wait for all tasks to finish
    pool.close()
    pool.join()

    return st, inv

#==================================================================================================

def get_station_inventory(
    starttime, endtime,
    client_list = ['IRIS', 'NCEDC', 'SCEDC'],
    channels_string = 'HH*,BH*,HN*,EH*',
    invetory_type="stationxml",
    minlatitude=39, maxlatitude=42, minlongitude=-126, maxlongitude=-122.5,
    ):

    """ 
    get the station inventory for a specific event from a list of clients with parallel processing.

    output:
           station inventory: stationtxt or stationxml
    """

        # create an empty inventory object
    merged_inventory = Inventory()


    # Loop through each client (IRIS, NCEDC, SCEDC data centers)
    for client_name in client_list:
        client = Client(client_name, debug=False, timeout=60)
        try:
            inv = client.get_stations(
                network="*",
                station="*",
                location="*",
                channel=channels_string,
                starttime=starttime,
                endtime=endtime,
                level="channel",
                minlatitude=minlatitude,
                maxlatitude=maxlatitude,
                minlongitude=minlongitude,
                maxlongitude=maxlongitude,

            )
            merged_inventory.networks.extend(inv.networks)
            # print(f"client.get_stations({client_name}) is successful")
            
        except Exception as e:
            # print(f"Error fetching data from {client_name}: {e}") 
            pass
    
    # return merged_inventory
    return merged_inventory


