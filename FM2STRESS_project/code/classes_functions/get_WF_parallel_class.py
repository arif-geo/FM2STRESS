import os
import io
import numpy as np
import pandas as pd

import obspy
from obspy.clients.fdsn import Client
from obspy import read, read_inventory, Stream, Inventory, UTCDateTime
from obspy.io.stationtxt.core import inventory_to_station_text

from tqdm import tqdm
# from multiprocessing import Pool
import multiprocessing

# pool = Pool(4)
# pool.apply_async


class GetWFInvParallel():
    def __init__(self, starttime, endtime, 
                    client_list = ['IRIS', 'NCEDC'],#, 'SCEDC'],
                    priority_channels_list = ['HHZ', 'BHZ', 'HNZ', 'EHZ'],
                    output_folder = None,
                    ):
        self.starttime = starttime
        self.endtime = endtime
        self.client_list = client_list
        self.priority_channels_list = priority_channels_list
        self.output_folder = output_folder
        self.success_stn = []

# ---------------------------- Get Inventory ----------------------------

    def get_inventory(self,
        radial_search: bool = True,
        event_lat: float = None,
        event_lon: float = None,
        minrad_ded: float = 0.0,
        maxrad_deg: float = 2.0,
        minlatitude=39,
        maxlatitude=42,
        minlongitude=-126,
        maxlongitude=-122.5,
        inv_path = None,   
    ):
      
        """

        """

        channels_string = ",".join(self.priority_channels_list)
        merged_inv = Inventory()
        for client_name in self.client_list:
            client = Client(client_name, debug=False, timeout=60)
            
            if radial_search == False:
                try:
                    inv = client.get_stations(
                        network="*",
                        station="*",
                        location="*",
                        channel=channels_string,
                        starttime=self.starttime,
                        endtime=self.endtime,
                        level="channel",
                        minlatitude=minlatitude,
                        maxlatitude=maxlatitude,
                        minlongitude=minlongitude,
                        maxlongitude=maxlongitude,
                    )
                    merged_inv.extend(inv.networks) 
                except Exception as e:
                    pass
            elif radial_search == True:
                try:
                    inv = client.get_stations(
                        network="*",
                        station="*",
                        location="*",
                        channel=channels_string,
                        starttime=self.starttime,
                        endtime=self.endtime,
                        level="channel",
                        latitude=event_lat,
                        longitude=event_lon,
                        minradius=minrad_ded,
                        maxradius=maxrad_deg,
                    )
                    merged_inv.extend(inv.networks) 
                except Exception as e:
                    pass
        merged_inv.write('./temp_inv.txt', format='stationtxt')
        invdf = pd.read_csv('./temp_inv.txt', sep='|', header=0)
        os.remove('./temp_inv.txt')
        invdf.sort_values(by=['#Network', 'Station'], inplace=True)
        invdf = invdf.drop_duplicates(subset=['#Network', 'Station'])
        
        if inv_path is not None:
            invdf.to_csv(inv_path, sep='|', index=False)

        return invdf

# ---------------------------- Get Waveforms [Parallel] ----------------------------
    
    def get_waveforms_parallel(self, 
        inventory_txt_path,
    ):

        st = Stream()
        inv = Inventory()

        # args list
        args_list = []

        input_inv = read_inventory(inventory_txt_path, format='stationtxt')

        for client_name in self.client_list:
            client = Client(client_name)

            for network in input_inv.networks:
                for station in network.stations:
                    args_list.append((client, network, station, self.priority_channels_list, self.starttime, self.endtime))

        # parallel processing
        pool = multiprocessing.Pool(os.cpu_count()-1)
        for temp_st, temp_inv in pool.imap_unordered(self.download_station, args_list):
            st += temp_st
            inv.extend(temp_inv.networks)

        pool.close()
        pool.join()

        # clean up
        for tr in st:
            if abs(tr.data.max()) == 0:
                st.remove(tr)

        return st, inv

# ---------------------------- Download Station [single] ----------------------------

    def download_station(self, args):

        # unpack args
        client, network, station, priority_channels_list, starttime, endtime = args
        st_local = Stream()
        inv_local = Inventory()

        for priority_channel in priority_channels_list:
            ch_list = list(set([ch.code[0:] for ch in station.channels]))

            if priority_channel[0:] not in ch_list:
                continue

            if station.code in self.success_stn:
                break # skip to next station

            try:
                temp_st = client.get_waveforms(
                    network=network.code,
                    station=station.code,
                    location="*",
                    channel=priority_channel,
                    starttime=starttime,
                    endtime=endtime,
                )
                
                if len(temp_st) == 0: 
                    print(temp_st)
                    continue 
                st_local += temp_st
                self.success_stn.append(station.code)
                break

            except Exception as e:
                pass

        return st_local, inv_local


# ---------------------------------------------------------------------------------

# ---------------------------- Class: GetWF New method ----------------------------

# ---------------------------------------------------------------------------------
class GetWF_Circular():
    def __init__(
        self, 
        client_list = ["NCEDC", "IRIS"],

    ):
        self.client_list = client_list


# ---------------------------- Get Waveforms [Single] ----------------------------

    def GetWF1client(self, client_name, row, channels_string):
        client = Client(client_name)
        inventory = client.get_stations(
            network='*',
            station='*',
            location='*',
            channel=channels_string,
            latitude=row.elat,
            longitude=row.elon,
            # approx. 200 km radius
            maxradius=1.5, #degree
            starttime=UTCDateTime(row.etime) - 30,
            endtime=UTCDateTime(row.etime) + 110,
            level='channel',
        )
        
        stations = []
        for net in inventory.networks:
            for stn in net:
                stations.append(stn.code)
        valid_channels = list(set([i[-3:] for i in inventory.get_contents()['channels']]))
        
        print(inventory.get_contents()['networks'])
        print(stations)
        print(valid_channels)

        # inventory.plot(projection='local', resolution='i')
        st = client.get_waveforms(
            network=",".join(inventory.get_contents()['networks']),
            station=",".join(stations),
            location="*",
            channel=",".join(valid_channels),
            starttime=UTCDateTime(row.etime) - 30,
            endtime=UTCDateTime(row.etime) + 120,
        )
        print('Download complete for ', client_name)

        return st, inventory   


# ---------------------------- Get Waveforms one event, all client ----------------------------
    def GetWF_1event(
        self, row, channels_string,

    ):
        stream_list = []
        inventory_list = []

        for cl in self.client_list:
            st, inv = self.GetWF1client(cl, row, channels_string)
            stream_list.append(st)
            inventory_list.append(inv)

        nctrid_list = [tr.id for tr in stream_list[0]]
        nc_data_len = len(stream_list[0])

        # merge the two streams, excluding the ones already in NCEDC
        counter = 0
        for tr in stream_list[1]:
            if tr.id not in nctrid_list:
                stream_list[0].append(tr)
                counter += 1
        
        # cleanup mseed file
        stream_list[0].resample(100)
        for tr in stream_list[0]:
            if len(tr.data) < 0.95 * 100 * 150: # 150 seconds of data, 100 Hz
                stream_list[0].remove(tr)
        
        # cleaned stationxml file
        inventory_list[0].extend(inventory_list[1])
        inventory_list[0].write('./temp.txt', 'STATIONTXT')
        inv = pd.read_csv('./temp.txt', delimiter='|').drop_duplicates(subset=['#Network', 'Station', 'Location', 'Channel'])
        inv.to_csv('./temp.txt', sep='|', header=True, index=False)
        inv = obspy.read_inventory('./temp.txt', format='STATIONTXT')
        os.remove('./temp.txt')

        return stream_list[0], inv

