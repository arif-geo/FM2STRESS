import os
import numpy as np
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import read, UTCDateTime, Stream, Inventory
from geopy.distance import geodesic
import tensorflow as tf
from keras import backend as K # for custom loss function
import matplotlib.pyplot as plt

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
#                                   WAVEFORM PLOTTER CLASS
# ==================================================================================================

class WaveformPlotter:
    """

    """
    def __init__(self,
                 mseed_file=None,
                 pol_df=None,
                 waveform_dir=None
                 ):
        self.mseed_file = mseed_file
        self.pol_df = pol_df
        self.waveform_dir = waveform_dir

    def WFprocessing(self,
        stn_id : str,
        pn_pick : UTCDateTime,
        st : Stream,
        sli : float = 0.5,
        normalize : bool = False,
    ):
        """ 
        Check for empty trace, detrend, taper, filter.
        Slice the trace around the PhasesNet pick.

        output: trace
        """ 
        ist = st.select(id=stn_id)
        tr = ist[0].copy()

        # check for empty trace
        if len(tr.data) > 0:
            tr.detrend('demean')
            try:
                tr.detrend('linear')
            except:
                tr.detrend('constant')
            
            try:
                tr.taper(0.001)
                tr.filter('bandpass', freqmin=.1, freqmax=10, corners=4, zerophase=True)  # Apply a bandpass filter
            except:
                pass
            
            if normalize == True:
                tr.data = tr.data/np.max(tr.data) # normalize the trace

            tr = tr.slice(pn_pick - sli-0.01, pn_pick + sli) # around PhasesNet pick

            return tr
    
    # ----------------------------------------------------------------------------------------------
    # END of WFprocessing function
    # ----------------------------------------------------------------------------------------------

    def plotWFpick_subplots(self,
                            mseed_file = None,
                            pol_df = None,
                            waveform_dir = None,
                            n_subplots=10, 
                            slice_len=0.5):
        
        """ 
        This plots each station waveform in a subplot with PhasesNet and Pyrocko picks.
        """ 


        # subset the df for this event only
        event_df = pol_df[pol_df['file_name'] == mseed_file]

        # check if the pyrocko col is empty or not [max should be 1]
        if event_df['pyrocko_polarity'].max() != 1:
            print(f'No pyrocko phase for {mseed_file}')
            return None

        ### cleanup the dataframe ###
        # drop rows with empty pyrocko phase_time [naT] and empty phasenet phase_time
        event_df = event_df.dropna(subset=['phase_time', 'pyrocko_phase_time']).reset_index(drop=True)

        # drop rows if time difference between phasenet and pyrocko is more than 2 seconds
        event_df['time_diff'] = (event_df['pyrocko_phase_time'] - event_df['phase_time']).dt.total_seconds()
        event_df = event_df[event_df['time_diff'].abs() <= 2].reset_index(drop=True)

        # read the mseed file
        st = read(f'{waveform_dir}/{mseed_file}')

        fig, axs = plt.subplots(n_subplots, 2, figsize=(10, 1.*n_subplots))

        for i, row in event_df[0:n_subplots].iterrows():

            pn_pick = UTCDateTime(pd.to_datetime(str(row['phase_time'])))
            pr_pick = UTCDateTime(pd.to_datetime(str(row['pyrocko_phase_time'])))
            
            stn_id = f'{row.station_id}Z'

            # use the WFprocessing function to get the trace
            tr = self.WFprocessing(stn_id, pn_pick, st, slice_len)

            if not len(tr.data) > 0: # check if the trace is empty
                continue

            # Generate time axis as a numpy array
            times = np.linspace(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.npts)

            # get the pick time from starttime into ms
            pn_pick_time = (pn_pick - tr.stats.starttime)
            pr_pick_time = (pr_pick - tr.stats.starttime)
            
            # get polarity
            dting_pol = row.diting_polarity
            dting_sharp = row.diting_sharpness
            prk_pol = row.pyrocko_polarity

            # easier subplot axes
            ax1, ax2 = axs[i, 0], axs[i, 1]


            ###### COLUMN 1: PhasesNet pick, DiTing polarity ######
            #######################################################
            # plot the waveform
            ax1.plot(times, tr.data, 'k') 

            # PhasesNet pick  
            ax1.axvline(x=pn_pick_time, color='r', linestyle='--')  

            # DiTing Polarity and sharpness
            ax1.text(
                pn_pick_time-0.2, tr.data.max()*0.5, 
                f"{dting_pol} / {dting_sharp}", 
                fontsize=12, color='b', ha='left')
            
            # Title [station_id] and horizontal line at 0
            ax1.set_title(
                f"{tr.id}",loc='left', 
                x=0.01, y=0.6, 
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, pad=0.1))

            # Horizontal line at 0
            ax1.axhline(y=0, color='k', lw=0.5)

            # hide x tick labels for this column except the last one
            if i != n_subplots-1:
                ax1.set_xticklabels([])
                ax1.set_xticks([])

            # ax1.spines['right'].set_visible(False)


            ###### COLUMN 2: Pyrocko pick, Pyrocko polarity ######
            #######################################################
            ax2.plot(times, tr.data, 'k')
            ax2.axvline(x=pr_pick_time, color='b', linestyle='--')           # Pyrocko pick [manual + phasenet]
            ax2.text(
                pn_pick_time-0.25, tr.data.max()*0.5,
                f"{'U' if prk_pol == 1 else 'D'}", 
                fontsize=12, color='r',) # polarity

            # horizontal line at 0
            ax2.axhline(y=0, color='k', lw=0.5)
            # hide x and y axis tick labels for this column
            ax2.set_yticklabels([])
            # move y axis ticks to the right
            ax2.yaxis.tick_right()
            if i != n_subplots-1:
                ax2.set_xticklabels([])
                ax2.set_xticks([])
            
            # # hide the left spine
            # ax2.spines['left'].set_visible(False)

        # add one title for each column
        axs[0, 0].set_title('PhasesNet and DiTingMotion', fontsize=14)
        axs[0, 1].set_title('Pyrocko and Phasenet', fontsize=14)

        # axis titles for the whole figure
        fig.text(0.5, -0.01, 'Time [s]', ha='center', fontsize=14)
        fig.text(-0.01, 0.5, 'Amplitude', va='center', rotation='vertical', fontsize=14)

        fig.subplots_adjust(wspace=0.1)
        fig.tight_layout()     

        return fig, axs

    # ----------------------------------------------------------------------------------------------
    # END of plotWFpick_subplots function
    # ----------------------------------------------------------------------------------------------

    def plotWFpick_oneplot(self,
                        mseed_file = None,
                        pol_df = None,
                        waveform_dir = None,
                        n_subplots=10, 
                        slice_len=0.5,
                        zoom=3,
                        normalize: bool = True,
                        hor_line: bool = True,
                        ):

        # subset the df for this event only
        event_df = pol_df[pol_df['file_name'] == mseed_file]

        # check if the pyrocko col is empty or not [max should be 1]
        if event_df['pyrocko_polarity'].max() != 1:
            print(f'No pyrocko phase for {mseed_file}')
            return None

        ### cleanup the dataframe ###
        # drop rows with empty pyrocko phase_time [naT] and empty phasenet phase_time
        event_df = event_df.dropna(subset=['phase_time', 'pyrocko_phase_time']).reset_index(drop=True)

        # drop rows if time difference between phasenet and pyrocko is more than 2 seconds
        event_df['time_diff'] = (event_df['pyrocko_phase_time'] - event_df['phase_time']).dt.total_seconds()
        event_df = event_df[event_df['time_diff'].abs() <= 2].reset_index(drop=True)

        # read the mseed file
        st = read(f'{waveform_dir}/{mseed_file}')

        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        ax1, ax2 = axs

        if n_subplots == 'all':
            n_subplots = len(event_df)
            
        for i, row in event_df[0:n_subplots].iterrows():

            # get the pick times (PhasesNet and Pyrocko)
            pn_pick = UTCDateTime(pd.to_datetime(str(row['phase_time'])))
            pr_pick = UTCDateTime(pd.to_datetime(str(row['pyrocko_phase_time'])))
            
            # get the trace for this station
            stn_id = f'{row.station_id}Z'

            # Process the waveform using the WFprocessing function
            tr = self.WFprocessing(stn_id, pn_pick, st, slice_len, normalize)
            
            if not len(tr.data) > 0: # check if the trace is empty
                continue
        
            # Generate time axis as a numpy array
            times = np.linspace(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.npts) # time [s]

            # get the pick time from starttime into seconds
            pn_pick_time = ((pn_pick - tr.stats.starttime)) 
            pr_pick_time = ((pr_pick - tr.stats.starttime)) 
            
            # get polarity
            dting_pol = row.diting_polarity
            dting_sharp = row.diting_sharpness
            prk_pol = row.pyrocko_polarity
            
            ###### COLUMN 1: PhasesNet pick, DiTing polarity ######
            #######################################################
            ax1.plot(times, tr.data*zoom + i, 'k')     # plot the waveform
            ax1.plot([pn_pick_time, pn_pick_time], [i-0.8, i+0.8], color='r', ls='--')      # PhasesNet pick
            # Polarity and sharpness (DiTing)
            ax1.text(
                pn_pick_time-0.15, i + 0.5, 
                f"{dting_pol} / {dting_sharp}", 
                fontsize=12, color='b', ha='left')

            # title [station_id] and horizontal line at 0
            ax1.text( 
                pn_pick_time-0.5,
                i - 0.3, 
                f"{tr.id}",
                fontsize=8)
                

            ###### COLUMN 2: Pyrocko pick, Pyrocko polarity ######
            #######################################################
            ax2.plot(times, tr.data * zoom +i, 'k')    # plot the waveform
            ax2.plot([pr_pick_time, pr_pick_time], [i-0.8, i+0.8], color='b', ls='--')   # Pyrocko pick [manual + phasenet]
            # Polarity (Pyrocko)
            ax2.text(
                pn_pick_time-0.25,  i + 0.5,
                f"{'U' if prk_pol == 1 else 'D'}", 
                fontsize=12, color='r',)

            ## Optional plotting parameters ##
            if hor_line:
                ax1.plot([0, 1], [i, i], color='k', lw=0.5)
                ax2.plot([0, 1], [i, i], color='k', lw=0.5)


        # # add one title for each column
        ax1.set_title('PhasesNet and DiTingMotion', fontsize=14)
        ax2.set_title('Pyrocko and Phasenet', fontsize=14)
        
        # plot axes (one for both columns)
        ax1.set_ylabel('Amplitude', fontsize=12)
        plt.xlabel('Time [s]',x=-.05, y=0.01, fontsize=12)

        return fig, axs

    # ----------------------------------------------------------------------------------------------
    #   END of plotWFpick_oneplot function
    # ----------------------------------------------------------------------------------------------


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
#                                       FILE CONVERTER CLASS
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
        # create new columns for pyrocko picks
        pndting_df['pyrocko_phase_time'], pndting_df['pyrocko_polarity'] = None, None

        # read pyrocko marker file
        with open(pyrocko_markers, 'r') as f:
            lines = f.readlines()[1:]                               # skip 1st line
            lines = lines[:-1] if len(lines[-1]) == 0 else lines    # remove the last line if it is empty
            total_lines = len(lines)

            current_event = None # key for the dictionary
            current_phases = []  # list as value for the dictionary
            event_data = {}      # dictionary to store the event data

            for i, line in enumerate(lines):
                line = line.strip().split()
                if line == lines[-1]:
                        print('last line:' , line)
                        event_data[current_event] = current_phases
                
                if line[0] == 'event:': # get event details
                    if current_event is not None:
                        # when a new event is found, store the previous event data
                        event_data[current_event] = current_phases
                        # print(current_event)
                        
                    current_event = line[-2]
                    current_phases = []       # reset the list for new event

                elif line[0] == 'phase:':
                    station_id_xx = line[4][:-1]
                    phase_time = line[1]+"T"+line[2]
                    phase_type = line[-3]
                    polarity = line[-2]
                    
                    # append the phase details to the list for the current event
                    current_phases.append([station_id_xx, phase_time, phase_type, polarity])

                    # if this is the last line, store the event data
                    if i == total_lines - 1:
                        event_data[current_event] = current_phases
                    

        for event_id, phase_data in event_data.items():
            evfile_name = f"{event_id}.mseed"

            for station_id, phase_time, phase_type, polarity in phase_data:

                # get the row index of the phasenet-diting dataframe 
                # by matching the event_id and station_id
                stn_rows = pndting_df[
                    (pndting_df.file_name == evfile_name) &
                    (pndting_df.station_id == station_id)
                    # & (pndting_df.phase_type == phase_type)
                    ].index
                                
                if len(stn_rows) > 0:
                    row_idx = stn_rows[0]
                    pndting_df.loc[row_idx, 'pyrocko_phase_time'] = phase_time
                    pndting_df.loc[row_idx, 'pyrocko_polarity'] = polarity

        return pndting_df #, event_data


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
        
