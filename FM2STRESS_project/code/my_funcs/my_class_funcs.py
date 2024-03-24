import os
import numpy as np
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import read, UTCDateTime, Stream, Inventory
from geopy.distance import geodesic
import tensorflow as tf
from keras import backend as K # for custom loss function

from tqdm.auto import tqdm
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import joblib 
import logging


class WaveformParallelDownloader:
    def __init__(self, 
                 starttime, endtime, 
                 minlon=-125.5, maxlon=-122, 
                 minlat=39.5, maxlat=42,
                 client_list=['IRIS', 'NCEDC'],  
                 priority_channels=['HH*', 'BH*', 'HN*', 'EH*']
                ):
        """
        Initialize the class with the following parameters:
        Mandatory arguments:
            starttime, endtime: UTCDateTime objects
        Optional arguments:
            minlon, maxlon, minlat, maxlat: float
            client_list: list of strings
            priority_channels: list of strings
        """

        self.starttime = starttime
        self.endtime = endtime
        self.minlon = minlon
        self.maxlon = maxlon
        self.minlat = minlat
        self.maxlat = maxlat
        self.client_list = client_list
        self.priority_channels = priority_channels
        self.success_stn = []

        # create a logger
        self.logger = logging.getLogger(__name__)                                              # create logger
        self.logger.setLevel(logging.INFO)                                                     # set logger level
        os.remove('waveform_parallel_downloader.log') if os.path.exists('waveform_parallel_downloader.log') else None
        file_handler = logging.FileHandler('waveform_parallel_downloader.log')                 # create file handler
        file_handler.setLevel(logging.INFO)                                                    # set file handler level
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # create formatter
        file_handler.setFormatter(formatter)                                                   # set formatter
        self.logger.addHandler(file_handler)                                                   # add file handler to logger


    def get_station_inventory(self, starttime, endtime):

        """
        Get station inventory for a specific time period.
        mandatory arguments:
            starttime, endtime: UTCDateTime objects

        optional arguments:
            minlon, maxlon, minlat, maxlat: float
            client_list: list of strings
            priority_channels: list of strings
        
        returns:
            inventory: obspy Inventory object
        """

        inventory = Inventory()

        for client_name in self.client_list:
            client = Client(client_name, debug=False, timeout=60)
            try:
                temp_inv = client.get_stations(
                    network="*", station="*", location="*",              
                    channel = ",".join(self.priority_channels), # convert ['HH*', 'BH*', 'HN*', 'EH*'] into 'HH*,BH*,HN*,EH*' [srting]
                    starttime=self.starttime, endtime=self.endtime,
                    minlatitude=self.minlat, maxlatitude=self.maxlat,
                    minlongitude=self.minlon, maxlongitude=self.maxlon,
                    level="channel"
                )
                # merge inventories
                inventory.networks.extend(temp_inv.networks)

            except Exception as e:
                self.logger.error(f"Error fetching data from {client_name}: {e}")
                continue

        return inventory


    # ==================================================================================================

    def get_waveforms_parallel(self):
        """
        Arranges all arguments for each station to download waveforms and distributes the work to each core.
        Actual download is done by `download_event_waveforms` function.
        """
        st = Stream()
        inv = Inventory() # stations with successful downloads

        # Get station inventory
        inventory = self.get_station_inventory(self.starttime, self.endtime)
        
        # Prepare arguments for download_station function
        args_list = []
        for client_name in self.client_list:
            client = Client(client_name, debug=False, timeout=30)

            for network in inventory.networks:
                for station in network.stations:
                    args_list.append((client, network, station, self.priority_channels, self.starttime, self.endtime, self.success_stn))

        # Parallelize the loop over stations
        #get core count
        core_count = os.cpu_count() - 1 # leave one core for other processes
        pool = Pool(core_count) # use all available cores

        with tqdm(total=len(args_list), disable=True) as pbar:
            for temp_st, temp_inv in pool.imap_unordered(self.download_station, args_list):
                pbar.update(1) # update progress bar for each station
                st += temp_st
                inv.networks.extend(temp_inv.networks)

        # Wait for all tasks to finish
        pool.close()
        pool.join()

        return st, inv        

    # ==================================================================================================

    def download_station(self, args):
        """
        Download waveforms for a single station.
        """
        client, network, station, priority_channels, starttime, endtime, success_stn = args
        st_local = Stream()
        inv_local = Inventory()

        for priority_channel in priority_channels:
            ch_list = list(set([ch.code[0:2] for ch in station.channels])) # list(set()) removes duplicates
            if priority_channel[0:2] not in ch_list:
                continue

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
                    channel=priority_channels,
                    starttime=starttime,
                    endtime=endtime,
                )

                temp_inv = client.get_stations(
                    network=network.code,
                    station=station.code,
                    location="*",
                    channel=priority_channels,
                    starttime=starttime,
                    endtime=endtime,
                    level="channel"
                )

                # if station.code not in success_stn:
                success_stn.append(station.code)
                st_local += temp_st
                inv_local.networks.extend(temp_inv.networks)

                # Break the loop if the right channel is found
                break

            except Exception as e:
                # self.logger.error(f"failed: {network.code}.{station.code}.{priority_channel}")
                pass
                

        return st_local, inv_local


# ==================================================================================================
# WAVEFORM PLOTTER CLASS
# ==================================================================================================

class WaveformPlotter:
    """

    """
    def __init__(self):
        pass
        
    def read_phasenet_event_picks(self):
        """
        Read the phasenet csv file and return a pandas dataframe with picks information
        for a single event.
        """
        # call a function from the MyFileReader class
        pass

    def get_event2station_distance(self):
        """
        Calculate the distance between the event and each station.
        """
        pass

    def plot_event_waveforms_dist(self):
        """
        Plot waveforms for each station for a single event.
        """
        pass

    def plot_FMP_pick(self):
        """
        Plot the FMP pick on the zoomed-in waveforms vs relative time [i.e. same start and end time for all stations]
        """
        pass


# ==================================================================================================
# FILE CONVERTER CLASS
# ==================================================================================================

class MyFileConverter:
    """
    Class to convert files from one format to another.
    """
    def __init__(self):
        pass

    def convert_pyrocko_picks_to_phasenet(self,
                                          pyrocko_markers,
                                          phasenet_diting_picks
                                          ):
        """
        Convert pyrocko picks to phasenet format.
        """
        # read phasenet picks
        pndting_df = pd.read_csv(phasenet_diting_picks, parse_dates=["begin_time", "phase_time"])

        # read pyrocko marker file
        with open(pyrocko_markers, 'r') as f:
            lines = f.readlines()[1:]        # skip 1st line

            current_event = None # key for the dictionary
            current_phases = []  # list as value for the dictionary
            event_data = {}      # dictionary to store the event data

            for line in lines:
                line = line.strip().split()
                
                if line[0] == 'event:': # get event details
                    if current_event is not None:
                        # make a new key with empty list values
                        event_data[current_event] = current_phases
                        
                    current_event = line[-2]
                    current_phases = []       # reset the list for new event

                elif line[0] == 'phase:':
                    station_id_xx = line[4][:-1]
                    phase_time = line[1]+"T"+line[2]
                    phase_type = line[-3]
                    polarity = line[-2]
                    
                    # append the phase details to the list for the current event
                    current_phases.append([station_id_xx, phase_time, phase_type, polarity])

        # print the dictionary keys

        event_data[list(event_data.keys())[0]]

        for event_id, phase_data in event_data.items():
            evfile_name = f"{event_id}.mseed"
            for station_id, phase_time, phase_type, polarity in phase_data:
                # get the row index of the phasenet-diting dataframe 
                # by matching the event_id and station_id
                row_idx = pndting_df[
                    (pndting_df.station_id == station_id) & 
                    (pndting_df.file_name==evfile_name) &
                    (pndting_df.phase_type == phase_type)
                    ].index
                if len(row_idx) > 0:
                    row_idx = row_idx[0]
                    pndting_df.loc[row_idx, 'pyrocko_phase_time'] = phase_time
                    pndting_df.loc[row_idx, 'pyrocko_polarity'] = polarity

        return pndting_df


# ==================================================================================================
# DITING-MOTION PICKER CLASS
# ==================================================================================================

class DitingMotionPicker:
    """
    Class to pick P and S waves from the diting-motion data.
    """
    def __init__(self, 
                    mseed_list=None,
                    waveform_dir=None,
                    phasenet_picks_df=None,
                    motion_model=None
                    ):
        
        self.mseed_list = mseed_list
        self.waveform_dir = waveform_dir
        self.phasenet_picks_df = phasenet_picks_df
        self.motion_model = motion_model
        self.phasenet_picks_df = self.phasenet_picks_df


    def assign_polarity(self, 
                        mseed_list,
                        waveform_dir,
                        phasenet_picks_df,
                        motion_model
                        ):
        """
        Pick P and S waves from the diting-motion data.
        Input:
            mseed_list: list with .mseed at the end of the file name
            waveform_dir: directory where the waveforms [.mseed] are stored
            phasenet_picks_df: pandas dataframe with phasenet picks
            motion_model: trained motion model [loaded using keras.models.load_model]

        Output:
            phasenet_picks_df: updated pandas dataframe with diting picks
        """
        # create output columns in the phasenet dataframe
        phasenet_picks_df['diting_polarity'], phasenet_picks_df['diting_sharpness'] = None, None
        
        # polarity dictionary [for details see below]
        polarity_dict = {0: 'U', 1: 'D'}
        sharpness_dict = {0: 'I', 1: 'E'}

        # create an empty array to store the motion input
        lensample = 128
        motion_input = np.zeros([1, lensample, 2]) # 1 trace, 128 samples, 2 components
        
        for mseed_file in (mseed_list):
            
            # read the mseed file
            st = read(f"{waveform_dir}/{mseed_file}")
            # phasenet picks for this event only
            temp_pn_df = phasenet_picks_df[phasenet_picks_df['file_name'] == mseed_file]

            for index, row in temp_pn_df.iterrows():
                stn_id_hh = row.station_id
                st_sel = st.select(id=f"{stn_id_hh}Z") # get the vertical component
                # print(tr_o)

                tr = st_sel[0].copy() # copy the trace
                if len(tr) > 0:
                    tr.detrend('demean') # demean the trace
                    try:
                        tr.detrend(type='linear') # remove the trend
                    except:
                        tr.detrend(type='constant') # remove the trend
                    
                    try:
                        tr.taper(0.001)
                        tr.filter('bandpass', freqmin=1, freqmax=20, corners=4, zerophase=True)
                    except:
                        pass 

                ##### now that we have a clean trace, we get the p_pick from the phasenet_df and cut the trace around the p_pick #####
                # get the PhaseNet pick time
                p_pick = UTCDateTime(pd.to_datetime(row.phase_time))

                lentime = lensample/100/2
                ### slice the trace around the p_pick
                tr = tr.slice(p_pick - 0.63, p_pick + 0.64) # 0.63 seconds before and 0.64 seconds after the p_pick
                # tr.plot()
                # break
                # print(len(tr))

                # check if the trace is the same length as the motion_input array
                if len(tr) == len(motion_input[0,:,0]):
                    motion_input[0,:,0] = tr.data[0:128] # assign the trace to the motion_input array
                
                # normalize the data by subtracting the mean and dividing by the standard deviation
                if np.max(motion_input[0,:,0]) != 0:
                    motion_input[0,:,0] -= np.mean(motion_input[0,:,0]) # subtract the mean to center the data
                    norm_factor = np.std(motion_input[0,:,0])

                    if norm_factor != 0:
                        motion_input[0,:,0] /= norm_factor # divide each element by the standard deviation
                        diff_data = np.diff(motion_input[0, 64:, 0])    # np.diff calculates the difference between each element
                        diff_sign_data = np.sign(diff_data)             # np.sign returns the sign of the difference
                        motion_input[0, 65:, 1] = diff_sign_data[:]     # assign the sign of the difference to the second component
                
                        ### PREDICT using Diting-Motion model
                        pred_res = motion_model.predict(motion_input, verbose=0) # prediction result as a dictionary
                        pred_fmp = (pred_res['T0D0'] + pred_res['T0D1'] + pred_res['T0D2'] + pred_res['T0D3']) / 4 # average of the 4 classes for the first motion prediction
                        pred_cla = (pred_res['T1D0'] + pred_res['T1D1'] + pred_res['T1D2'] + pred_res['T1D3']) / 4 # average of the 4 classes for CLA(idk)

                        """
                        if the 1st index is the maximum, then the polarity is UP/positive (+1)
                        or, if the 2nd index is the maximum, then the polarity is DOWN/negative (-1)
                        otherwise, the polarity is x [the 3rd index is the maximum, undecided (0/x)]
                        """
                        # test = np.argmax(pred_fmp[0, :])
                        # print(test)
                        polarity = polarity_dict.get(np.argmax(pred_fmp[0, :]), 'x')    # if max = 0, polarity = U, if max = 1, polarity = D, else polarity = x
                        sharpness = sharpness_dict.get(np.argmax(pred_cla[0, :]), 'x')  # if max = 0, sharpness = I, if max = 1, sharpness = E, else sharpness = x

                        # assign the polarity to the phasenet dataframe into a new column `diting_polarity` & `diting_sharpness`
                        phasenet_picks_df.loc[index, 'diting_polarity'] = polarity
                        phasenet_picks_df.loc[index, 'diting_sharpness'] = sharpness

        return phasenet_picks_df


    # ==================================================================================================
    # similar func but parallelized
    # ==================================================================================================

    def assign_polarity_parallel(self):

        self.phasenet_picks_df['diting_polarity'], self.phasenet_picks_df['diting_sharpness'] = None, None

        with ProcessPoolExecutor() as executor:
            temp_dfs = list(tqdm(
                        executor.map(
                                self.one_event_polarity,            # function to execute
                                self.mseed_list,                                # arg1
                                chunksize=1), # it means that each task will be executed in a separate process
                                total=len(self.mseed_list),
                                desc='Polarity Detection',
                                leave=False
                                ))
        
        # convert the returned lists of dataframes into a single dataframe
        new_pn_df = pd.concat(temp_dfs, ignore_index=True)
        return new_pn_df


    # ==================================================================================================

    def one_event_polarity(self, mseed_file):
        """ 
        
        """
        # phasenet picks for this event only
        temp_pn_df = self.phasenet_picks_df[self.phasenet_picks_df['file_name'] == mseed_file]

        # read the mseed file
        st = read(f"{self.waveform_dir}/{mseed_file}")

        # polarity dictionary [for details see below]
        polarity_dict = {0: 'U', 1: 'D'}
        sharpness_dict = {0: 'I', 1: 'E'}

        # create an empty array to store the motion input
        lensample = 128
        motion_input = np.zeros([1, lensample, 2]) # 1 trace, 128 samples, 2 components


        for index, row in temp_pn_df.iterrows():
            stn_id_hh = row.station_id
            st_sel = st.select(id=f"{stn_id_hh}Z") # get the vertical component

            tr = st_sel[0].copy() # copy the trace
            if len(tr) > 0:
                tr.detrend('demean') # demean the trace
                try:
                    tr.detrend(type='linear') # remove the trend
                except:
                    tr.detrend(type='constant') # remove the trend
                
                try:
                    tr.taper(0.001)
                    tr.filter('bandpass', freqmin=2, freqmax=10, corners=4, zerophase=True)
                except:
                    pass 

            ##### now that we have a clean trace, we get the p_pick from the phasenet_df and cut the trace around the p_pick #####
            # get the PhaseNet pick time
            p_pick = UTCDateTime(pd.to_datetime(row.phase_time))

            lentime = lensample/100/2
            ### slice the trace around the p_pick
            tr = tr.slice(p_pick-(lentime-0.01), p_pick+lentime) # 0.63 seconds before and 0.64 seconds after the p_pick

            if len(tr) == len(motion_input[0,:,0]):
                motion_input[0,:,0] = tr.data[0:128] # assign the trace to the motion_input array
            
                # normalize the data by subtracting the mean and dividing by the standard deviation
                if np.max(motion_input[0,:,0]) != 0:
                    motion_input[0,:,0] -= np.mean(motion_input[0,:,0]) # subtract the mean to center the data
                    norm_factor = np.std(motion_input[0,:,0])

                    if norm_factor != 0:
                        motion_input[0,:,0] /= norm_factor # divide each element by the standard deviation
                        diff_data = np.diff(motion_input[0, 64:, 0])    # np.diff calculates the difference between each element
                        diff_sign_data = np.sign(diff_data)             # np.sign returns the sign of the difference
                        motion_input[0, 65:, 1] = diff_sign_data[:]     # assign the sign of the difference to the second component
            
                        ### PREDICT using Diting-Motion model
                        pred_res = self.motion_model.predict(motion_input, verbose=0) # prediction result as a dictionary
                        pred_fmp = (pred_res['T0D0'] + pred_res['T0D1'] + pred_res['T0D2'] + pred_res['T0D3']) / 4 # average of the 4 classes for the first motion prediction
                        pred_cla = (pred_res['T1D0'] + pred_res['T1D1'] + pred_res['T1D2'] + pred_res['T1D3']) / 4 # average of the 4 classes for CLA(idk)

                        """
                        if the 1st index is the maximum, then the polarity is UP/positive (+1)
                        or, if the 2nd index is the maximum, then the polarity is DOWN/negative (-1)
                        otherwise, the polarity is x [the 3rd index is the maximum, undecided (0/x)]
                        """
                    
                        polarity = polarity_dict.get(np.argmax(pred_fmp[0, :]), 'x')    # if max 1st index, polarity = U, if max 2nd index, polarity = D, else polarity = x
                        sharpness = sharpness_dict.get(np.argmax(pred_cla[0, :]), 'x')  # if max 1st index, sharpness = I, if max 2nd index, sharpness = E, else sharpness = x
                        # print(polarity, sharpness)

                        # assign the polarity to the phasenet dataframe into the new columns
                        temp_pn_df.loc[index, 'diting_polarity'] = polarity
                        temp_pn_df.loc[index, 'diting_sharpness'] = sharpness
                        # print(index, polarity, sharpness)

        return temp_pn_df
        
