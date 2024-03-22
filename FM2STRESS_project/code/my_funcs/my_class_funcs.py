import os
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import read_inventory, UTCDateTime, Stream, Inventory
from tqdm import tqdm
from multiprocessing import Pool
import joblib 
import logging


class ClientWrapper:
    """
    A wrapper class to encapsulate an ObsPy FDSN client object.
    """

    def __init__(self, client_name):
        self.client = Client(client_name, debug=False, timeout=30)

    def get_waveforms(self, network, station, location="*", channel="*", starttime=None, endtime=None):
        """
        Downloads waveforms from a specific client for a given station and channel.

        Args:
            network (str): Network code.
            station (str): Station code.
            location (str, optional): Location code. Defaults to "*".
            channel (str, optional): Channel code. Defaults to "*".
            starttime (UTCDateTime, optional): Start time of the event. Defaults to None.
            endtime (UTCDateTime, optional): End time of the event. Defaults to None.

        Returns:
            Stream: A Stream object containing the downloaded waveforms.
        """

        try:
            return self.client.get_waveforms(network=network, station=station, location=location,
                                             channel=channel, starttime=starttime, endtime=endtime)
        except Exception as e:
            logging.error(f"Failed to download waveforms for {network}.{station}.{channel} from {self.client.base_url}: {e}")
            return Stream()  # Return an empty stream on errors


class EventDownloader:
    """
    A class to manage parallel waveform and station inventory download for an event.
    """

    def __init__(self, client_list, inventory, starttime, endtime, output_folder, priority_channels):
        """
        Initializes the EventDownloader object.

        Args:
            client_list (list): List of client names (e.g., ["IRIS", "SCEDC"]).
            inventory (Inventory): ObsPy Inventory object.
            starttime (UTCDateTime): Start time of the event.
            endtime (UTCDateTime): End time of the event.
            output_folder (str): Output folder for downloaded data.
            priority_channels (list): List of priority channels to download.
        """

        self.client_list = client_list
        self.inventory = inventory
        self.starttime = starttime
        self.endtime = endtime
        self.output_folder = output_folder
        self.priority_channels = priority_channels
        self.success_stn = []  # Track downloaded stations (internal state)

        # Set up logging
        logging.basicConfig(filename='get_waveforms_parallel.log', level=logging.INFO)

    def download_waveforms(self):
        """
        Downloads waveforms for the event in parallel using a process pool.
        """

        logging.info(f"Downloading waveforms...")

        # Prepare arguments for download_worker function
        args_list = []
        for client_name in self.client_list:
            client = ClientWrapper(client_name)
            for network in self.inventory.networks:
                for station in network.stations:
                    args_list.append((client, network, station, self.priority_channels, self.starttime, self.endtime))

        # Use a process pool for parallel downloads
        with Pool(processes=6) as pool:
            results = pool.starmap(self.download_worker, args_list)

        # Combine downloaded waveforms and inventory
        self.waveforms, self.inv = zip(*results)
        self.waveforms = Stream.concatenate(*self.waveforms)
        self.inv = Inventory()
        for inv in self.inv:
            self.inv.networks.extend(inv.networks)

        logging.info(f"Download complete. Downloaded {len(self.success_stn)} stations out of {len(self.inventory.get_contents()['stations'])}")

    def download_worker(self, client, network, station, priority_channels, starttime, endtime):
        """
        Downloads waveforms and station inventory for a single station (worker function for the pool).

        Args:
            client (ClientWrapper): ClientWrapper object.
            network (Network): Network object from the inventory

        Returns:
            Stream: A Stream object containing the downloaded waveforms.
            Inventory: An Inventory object containing the downloaded station inventory.
        """

        st_local = Stream()
        inv_local = Inventory()

        for priority_channel in priority_channels:
            ch_list = list(set([ch.code[0:2] for ch in station.channels]))  # list(set()) removes duplicates
            if priority_channel[0:2] not in ch_list:
                continue

            try:
                temp_st = client.get_waveforms(network.code, station.code, "*", priority_channel, starttime, endtime)
                temp_inv = client.client.get_stations(network=network.code, station=station.code, starttime=starttime, endtime=endtime)

                if len(temp_st) > 0:
                    self.success_stn.append(station.code)
                    st_local += temp_st
                    inv_local.networks.extend(temp_inv.networks)
                    logging.info(f"Waveform downloaded: {network.code}.{station.code}.{priority_channel} from {client.client.base_url}.")
                    logging.info(f"Inventory downloaded: {len(self.success_stn)} out of {len(self.inventory.get_contents()['stations'])}")
                    break
            except Exception as e:
                logging.error(f"Failed to download station information for {network.code}.{station.code}.{priority_channel} from {client.client.base_url}: {e}")

        return st_local, inv_local
