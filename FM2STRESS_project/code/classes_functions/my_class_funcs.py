import os
import numpy as np
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import read, UTCDateTime, Stream, Inventory
from geopy.distance import geodesic
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K # for custom loss function
# from tensorflow.keras import backend as K

from tqdm.auto import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import logging
import pdb


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
    
    # ----------------------------------------------------------------------------------------------
    # WFprocessing function

    def WFprocessing(self,
        stn_id : str,
        pn_pick : UTCDateTime,
        st : Stream,
        sli : float = 0.5,
        freqmin=1.5, 
        freqmax=10,
        normalize : bool = False,
    ):
        """ 
        Check for empty trace, detrend, taper, filter.
        Slice the trace around the PhasesNet pick.

        output: trace
        """ 
        ist = st.select(id=stn_id)
        ist.resample(100) # resample to 100 Hz
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
                tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)  # Apply a bandpass filter
            except:
                pass
            
            if normalize == True:
                tr.data = tr.data/np.max(tr.data) # normalize the trace

            tr = tr.slice(pn_pick - (sli-0.01), pn_pick + sli) # around PhasesNet pick

            return tr

    # .............................. SUB-FUNCTION .................................................
    # ................. process_df_subset function ...............................................

    def process_df_subset(self, mseed_file, pol_df, sort_by=None, pyrocko_col=True):

        """
        Process the dataframe for a single event.
        input: 
            mseed_file: str
            pol_df: pandas dataframe
            sort_by: str [optional]
                default: None
                options: PhaseNet file column names['phase_score', etc.] 
        output:
            event_df: pandas dataframe
        
        N.B. To be used within the plotter functions.
                     
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

        if sort_by is not None:
            # sort by phase_score for the best picks
            event_df = event_df.sort_values(by='phase_score', ascending=False)

        return event_df

    # . . . . . . . . . . . . . . SUB-FUNCTION . . . . . . . . . . . . . . . . . . . . . . . . 
    # . . . . . . . . get_trace2plot function ...............................................

    def get_trace2plot(self, row, st, slice_len, normalize,
                        ):
        """
        Prepare the trace for plotting with PhasesNet and Pyrocko picks and polarities.
            To be used within the plotter functions. 
        """

        # get the pick times (PhasesNet and Pyrocko)
        pn_pick = UTCDateTime(pd.to_datetime(str(row['phase_time'])))
        pr_pick = UTCDateTime(pd.to_datetime(str(row['pyrocko_phase_time'])))
        
        # get the trace for this station
        stn_id = f'{row.station_id}Z'

        # Process the trace using the WFprocessing function
        tr = self.WFprocessing(stn_id, pn_pick, st, slice_len, normalize)
        
        if not len(tr.data) > 0: # check if the trace is empty
            return None, None, None, None, None, None, None
    
        # Generate time axis as a numpy array
        times = np.linspace(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.npts) # time [s]

        # get the pick time from starttime into seconds
        pn_pick_time = ((pn_pick - tr.stats.starttime)) 
        pr_pick_time = ((pr_pick - tr.stats.starttime)) 
        
        # get polarity
        dting_pol, dting_sharp = row.diting_polarity, row.diting_sharpness
        prk_pol = row.pyrocko_polarity

        return tr, times, pn_pick_time, pr_pick_time, dting_pol, dting_sharp, prk_pol


    # ----------------------------------------------------------------------------------------------
    # plotWFpick_subplots function
    # ----------------------------------------------------------------------------------------------

    def plotWFpick_subplots(self,
                            mseed_file = None,
                            pol_df = None,
                            waveform_dir = None,
                            figsize=(10, 6),
                            n_subplots=10, 
                            slice_len=0.5,
                            normalize: bool = True,
                            hide_spines: bool = False,
                            hor_line: bool = True,
                            ):
        
        """ 
        This plots each station waveform in a subplot with PhasesNet and Pyrocko picks.
        """ 

        # subset the df for this event only
        event_df = self.process_df_subset(mseed_file, pol_df, sort_by=None)

        # read the mseed file
        st = read(f'{waveform_dir}/{mseed_file}')

        fig, axs = plt.subplots(n_subplots, 2, figsize=figsize)

        for i, row in event_df[0:n_subplots].iterrows():
            # prepare the trace for plotting with PhasesNet and Pyrocko picks and polarities
            tr, times, pn_pick_time, pr_pick_time, dting_pol, dting_sharp, prk_pol = self.get_trace2plot(row, st, slice_len, normalize)

            # easier subplot axes
            ax1, ax2 = axs[i, 0], axs[i, 1]

            ###### COLUMN 1: PhasesNet pick, DiTing polarity ######
            #######################################################
            # plot the waveform
            ax1.plot(times, tr.data, 'k', lw=0.5) 

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
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, pad=0.1),
                alpha=0.8)

            # hide x tick labels for this column except the last one
            if i != n_subplots-1:
                ax1.set_xticklabels([])
                ax1.set_xticks([])


            ###### COLUMN 2: Pyrocko pick, Pyrocko polarity ######
            #######################################################
            ax2.plot(times, tr.data, 'k', lw=0.5) # plot the waveform
            ax2.axvline(x=pr_pick_time, color='b', linestyle='--')           # Pyrocko pick [manual + phasenet]
            ax2.text(
                pn_pick_time-0.25, tr.data.max()*0.5,
                int(prk_pol) if not np.isnan(prk_pol) else prk_pol,
                fontsize=12, color='r',) # polarity

            # move y axis ticks to the right
            ax2.yaxis.tick_right()

            # hide y axis tick labels for this column
            ax2.set_yticklabels([])

            # hide x tick labels for this column except the last one
            if i != n_subplots-1: 
                ax2.set_xticklabels([])
                ax2.set_xticks([])


        # Customize all the subplots
           
        # add one title for each column
        axs[0, 0].set_title('PhasesNet and DiTingMotion', fontsize=14)
        axs[0, 1].set_title('Pyrocko Handpicked', fontsize=14)

        # axis titles for the whole figure
        fig.text(0.5, 0.04, 'Time [s]', ha='center', fontsize=14)
        fig.text(0.04, .5, f'{"Normalized" if normalize==True else ""} Amplitude', va='center', rotation='vertical', fontsize=14)
        
        axs = axs.flatten()
        # # hide axis spines
        if hide_spines:
            for i, ax in enumerate(axs):
                ax.spines['top'].set_visible(False) if i != 0 and i != 1 else None
                ax.spines['right'].set_visible(False) if i % 2 == 0 else None
                ax.spines['bottom'].set_visible(False) if ax != axs[-1] and ax != axs[-2] else None
                # ax.spines['left'].set_visible(False) if i % 2 == 1 else None
        
        # horizontal line at 0
        if hor_line:
            for ax in axs:
                ax.axhline(y=0, color='k', lw=0.5)
        # hide y axis labels for all subplots
        plt.setp(axs, yticks=[])

        # fig.subplots_adjust(wspace=0.1)
        # fig.tight_layout()     

        return fig, axs

    # ----------------------------------------------------------------------------------------------
    # plotWFpick_oneplot function
    # ----------------------------------------------------------------------------------------------

    def plotWFpick_oneplot(self,
                        mseed_file = None,
                        pol_df = None,
                        waveform_dir = None,
                        figsize=(10, 6),
                        n_subplots=10, 
                        slice_len=0.5,
                        zoom=5,
                        normalize: bool = True,
                        hor_line: bool = True,
                        
                        ):

        # subset the df for this event only
        event_df = self.process_df_subset(mseed_file, pol_df, sort_by=None) 

        # read the mseed file
        st = read(f'{waveform_dir}/{mseed_file}')

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax1, ax2 = axs

        if n_subplots == 'all' or n_subplots > len(event_df):
            n_subplots = len(event_df)


        for i, row in event_df[0:n_subplots].iterrows():
            # prepare the trace for plotting with PhasesNet and Pyrocko picks and polarities
            tr, times, pn_pick_time, pr_pick_time, dting_pol, dting_sharp, prk_pol = self.get_trace2plot(row, st, slice_len, normalize)

            if tr is None: # skip if the trace is empty
                continue

            # variable zoom using spread of the data
            spread = np.std(tr.data)          
            vzoom = zoom/np.round(spread, 2)/10 if spread != 0 else zoom
            
            ###### COLUMN 1: PhasesNet pick, DiTing polarity ######
            #######################################################

            ax1.plot(times, tr.data * vzoom + i, 'k', lw=0.5)     # plot the waveform
            ax1.plot([pn_pick_time, pn_pick_time], [i-0.4, i+0.4], color='r', ls='--')      # PhasesNet pick
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
            ax2.plot(times, tr.data * vzoom +i, 'k', lw=0.5)    # plot the waveform
            ax2.plot([pr_pick_time, pr_pick_time], [i-0.4, i+0.4], color='b', ls='--')   # Pyrocko pick [manual + phasenet]
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
    

# ==================================================================================================
# DITING-MOTION PICKER CLASS
# ==================================================================================================

class DitingMotionPicker:
    """
    Class run the Diting-Motion model to assign polarity to the P-wave picks.
    """

    def __init__(self, 
                    mseed_list=None,
                    waveform_dir=None,
                    phasenet_picks_df=None,
                    motion_model=None,
                    lensample=128,
                    correct_pn_picks=0.0
                    ):
        
        self.mseed_list = mseed_list
        self.waveform_dir = waveform_dir
        self.phasenet_picks_df = phasenet_picks_df
        self.motion_model = motion_model
        self.lensample = lensample
        self.correct_pn_picks = correct_pn_picks
        # self.phasenet_picks_df = self.phasenet_picks_df 


    # ..............................................................................................
    #                   ASSIGN POLARITY USING DITING-MOTION MODEL
    # ..............................................................................................

    def assign_polarity(self, 
                        
                        ):
        """
        Assign polarity to the PhaseNet picks using the Diting-Motion model.
        Input:
            mseed_list: list with .mseed at the end of the file name
            waveform_dir: directory where the waveforms [.mseed] are stored
            phasenet_picks_df: pandas dataframe with phasenet picks
            motion_model: trained motion model [loaded using keras.models.load_model]

        Output:
            phasenet_picks_df: updated pandas dataframe with diting picks
        """
        mseed_list = self.mseed_list
        waveform_dir = self.waveform_dir
        phasenet_picks_df = self.phasenet_picks_df
        motion_model = self.motion_model

        # create output columns in the phasenet dataframe
        phasenet_picks_df['diting_polarity'], phasenet_picks_df['diting_sharpness'] = None, None
        
        # polarity dictionary [for details see below]
        polarity_dict = {0: '1', 1: '-1'} #{0: 'U', 1: 'D'}
        sharpness_dict = {0: 'I', 1: 'E'} #{0: 'I', 1: 'E'}

        # create an empty array to store the motion input
        lensample = self.lensample
        motion_input = np.zeros([1, lensample, 2]) # 1 trace, 128 samples, 2 components
        
        pbar = tqdm(total=len(mseed_list), desc='Polarity Detection')
        counta = 0
        countb = 0
        for mseed_file in (mseed_list):
            
            # read the mseed file
            st = read(f"{waveform_dir}/{mseed_file}")
            st = st.resample(100)

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
                        tr.filter('bandpass', freqmin=1.5, freqmax=10, corners=4, zerophase=True)
                    except:
                        pass 

                ##### now that we have a clean trace, we get the p_pick from the phasenet_df and cut the trace around the p_pick #####
                # get the PhaseNet pick time
                p_pick = UTCDateTime(pd.to_datetime(row.phase_time)) + self.correct_pn_picks

                lentime = lensample/100/2
                ### slice the trace around the p_pick
                tr = tr.slice(p_pick - lentime, p_pick + lentime) # 0.64s before and after the p_pick (1.28s total)
                
                # Due to decimal rounding, the trace length may be 129 instead of 128.
                # check if the trace is the same length as the motion_input array
                if len(tr.data) >= 128:
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
                        pred_res = motion_model.predict(
                            motion_input, 
                            # verbose=0,
                            ) # prediction result as a dictionary
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
            # break
            pbar.update(1)
        print(f"tr = 128: {counta}, tr = 127: {countb}")
        return phasenet_picks_df


    # ----------------------------------------------------------------------------------------------
    # similar func but parallelized
    # ----------------------------------------------------------------------------------------------

    def assign_polarity_parallel(self,
                                mseed_list=None,
                                waveform_dir=None,
                                phasenet_picks_df=None,
                                motion_model=None
                                ):
        """
        Assign polarity to the PhaseNet picks using the Diting-Motion model.
        Parallelized version.
        """
        # create output columns in the phasenet dataframe
        self.phasenet_picks_df['diting_polarity'], self.phasenet_picks_df['diting_sharpness'] = None, None

        with multiprocessing.Pool(os.cpu_count()-1) as pool:
            print(f"Running parallel processing with {os.cpu_count()-1} cores")
            temp_dfs = list(tqdm(
                pool.imap_unordered(
                        self.one_event_polarity,            # function to execute
                        self.mseed_list,                                # arg1
                        chunksize=1 # it means that each task will be executed in a separate process
                        ),
                    total=len(self.mseed_list),
                    desc='Polarity Detection',
                    leave=True # progress bar leaves when done
                    )
                )
        
        # convert the returned lists of dataframes into a single dataframe
        new_pn_df = pd.concat(
            temp_dfs, ignore_index=True
            ).sort_values(by=['file_name', 'phase_score']) #, 'sta_dist_km'
        
        return new_pn_df


    # ==================================================================================================

    def one_event_polarity(self, mseed_file):
        """ 
        Predict polarity for a single event using the Diting-Motion model.
        """
        # phasenet picks for this event only
        temp_pn_df = self.phasenet_picks_df[self.phasenet_picks_df['file_name'] == mseed_file]

        # read the mseed file
        st = read(f"{self.waveform_dir}/{mseed_file}")

        # resample the trace to 100 Hz
        st.resample(100)

        # polarity dictionary [for details see below]
        polarity_dict = {0: '1', 1: '-1'}
        sharpness_dict = {0: 'I', 1: 'E'}

        # create an empty array to store the motion input
        lensample = self.lensample
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
                    tr.filter('bandpass', freqmin=1.5, freqmax=10)      #, corners=4, zerophase=True)
                except:
                    pass

            ##### now that we have a clean trace, we get the p_pick from the phasenet_df and cut the trace around the p_pick #####
            # get the PhaseNet pick time
            p_pick = UTCDateTime(pd.to_datetime(row.phase_time)) + self.correct_pn_picks

            ### slice the trace around the p_pick
            lentime = self.lensample/100/2
            tr = tr.slice(p_pick-lentime, p_pick+lentime)               # 0.6 seconds before and after the p_pick
            
            # Due to decimal rounding, the trace length may be 129 instead of 128.  
            # check if the trace is the same length as the motion_input array
            if len(tr.data) >= 128:
                motion_input[0,:,0] = tr.data[0:128] # assign the trace to the motion_input array

                # normalize the data by subtracting the mean and dividing by the standard deviation
                if np.max(motion_input[0,:,0]) != 0:                    # check if the maximum value is not zero
                    motion_input[0,:,0] -= np.mean(motion_input[0,:,0]) # subtract the mean to center the data
                    norm_factor = np.std(motion_input[0,:,0])

                    if norm_factor != 0:
                        motion_input[0,:,0] /= norm_factor              # divide each element by the standard deviation
                        diff_data = np.diff(motion_input[0, 64:, 0])    # np.diff calculates the difference between each element
                        diff_sign_data = np.sign(diff_data)             # np.sign returns the sign of the difference
                        motion_input[0, 65:, 1] = diff_sign_data[:]     # assign the sign of the difference to the second component

                        ### PREDICT using Diting-Motion model
                        # prediction result as a dictionary
                        pred_res = self.motion_model.predict(
                            motion_input, 
                            verbose=0,
                            # experimental_relax_shapes=True,
                            ) 
                        pred_fmp = (pred_res['T0D0'] + pred_res['T0D1'] + pred_res['T0D2'] + pred_res['T0D3']) / 4 # average of the 4 classes for the first motion prediction
                        pred_cla = (pred_res['T1D0'] + pred_res['T1D1'] + pred_res['T1D2'] + pred_res['T1D3']) / 4 # average of the 4 classes for CLA(idk)
                        # print(pred_fmp[0, :])

                        """
                        if the 1st index is the maximum, then the polarity is UP/positive (+1)
                        or, if the 2nd index is the maximum, then the polarity is DOWN/negative (-1)
                        otherwise, the polarity is x [the 3rd index is the maximum, undecided (0/x)]
                        """

                        polarity = polarity_dict.get(np.argmax(pred_fmp[0, :]), 'x')    # if max 1st index, polarity = U, if max 2nd index, polarity = D, else polarity = x
                        sharpness = sharpness_dict.get(np.argmax(pred_cla[0, :]), 'x')  # if max 1st index, sharpness = I, if max 2nd index, sharpness = E, else sharpness = x
                        # print(polarity, sharpness)
                        # print(np.argmax(pred_fmp[0, :]))

                        # assign the polarity to the phasenet dataframe into the new columns
                        temp_pn_df.loc[index, 'diting_polarity'] = polarity
                        temp_pn_df.loc[index, 'diting_sharpness'] = sharpness
                        # print(index, polarity, sharpness)

        return temp_pn_df


# ==================================================================================================
#                                       FILE CONVERTER CLASS
# ==================================================================================================

class MyFileConverter:
    """
    Class to convert files from one format to another.
    """
    def __init__(self):
        pass
    
    # ..............................................................................................
    #                   PYROCKO MARKER FILE TO PHASENET PICKS [VERSION 2]
    # ..............................................................................................

    # .................. SUB-FUNCTION .....................
    def PyrockoMarker2Dict(self, pyrocko_markers):
        """
        Read a Pyrocko marker file and convert to a dictionary.
        output:
        {
            event_id: [[station_id, phase_time, phase_type, polarity], ...],
            ...
        }
        """
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
                
                if line[0] == 'event:': # get event details
                    if current_event is not None:
                        # when a new event is found, store the previous event data
                        event_data[current_event] = current_phases
                        # print(current_event)
                        
                    current_event = line[-2]
                    current_phases = []       # reset the list for new event

                elif line[0] == 'phase:':
                    phase_type = line[-3]
                    if phase_type != 'P':
                        continue
                    station_id_xx = line[4][:-1]
                    phase_time = line[1]+"T"+line[2]
                    polarity = line[-2]
                    
                    # append the phase details to the list for the current event
                    current_phases.append([station_id_xx, phase_time, phase_type, polarity])

                    # if this is the last line, store the event data
                    if i == total_lines - 1:
                        event_data[current_event] = current_phases

        return event_data # dictionary with event_id as key and list of phases as value

    # ..............................................................................................
    def convert_pyrocko_picks_to_phasenet(self, pyrocko_markers, phasenet_diting_picks):
        """
        Convert pyrocko picks to phasenet format.
        input: 
            pyrocko_markers, 
            phasenet_diting_picks
        
        N.B.: Only adds the pyrocko picks to the phasenet dataframe.
              Does not create a new dataframe.
        """

        # read phasenet picks
        pndting_df = pd.read_csv(phasenet_diting_picks, parse_dates=["begin_time", "phase_time"])

        # create new columns for pyrocko picks
        pndting_df['pyrocko_phase_time'], pndting_df['pyrocko_polarity'] = None, None
                    
        new_row_count, skipped_row_count = 0, 0

        # read pyrocko marker file and convert to dictionary
        event_data = self.PyrockoMarker2Dict(pyrocko_markers)
        
        for event_id, phase_data in event_data.items():
            evfile_name = f"{event_id}.mseed"

            # subset the dataframe for this event only
            event_df = pndting_df[pndting_df['file_name'] == evfile_name]

            for station_id_xx, phase_time, phase_type, polarity in phase_data:
                polarity = str(polarity)
                # print(type(station_id_xx), type(evfile_name), type(phase_type), type(phase_time), type(polarity))
                # Efficiently find matching rows
                match_mask = (event_df['station_id'] == station_id_xx) & \
                            (event_df['phase_type'] == phase_type)  # Optional matching

                matching_rows = event_df[match_mask].sort_values(by='phase_type')   # P and then S

                if not matching_rows.empty:                     # At least one matching row exists 
                    row_idx = matching_rows.index[0]            # Get the first matching row
                    pndting_df.loc[row_idx, 'pyrocko_phase_time'] = phase_time
                    pndting_df.loc[row_idx, 'pyrocko_polarity'] = polarity

                # If no match, but polarity is valid and phase_type is P, add a new row
                elif polarity in ['-1', '1'] and phase_type == 'P':
                    new_row_dict = {
                        'file_name': evfile_name,
                        'station_id': station_id_xx,
                        'phase_type': phase_type,
                        'pyrocko_phase_time': phase_time,
                        'pyrocko_polarity': polarity,
                    }
                    # print(new_row_dict) 
                    new_row = pd.DataFrame([new_row_dict])

                    # add the new row to the dataframe
                    if not new_row.empty and not new_row.isna().all().all(): # Check if the new row is not empty
                        pndting_df = pd.concat([pndting_df, new_row], ignore_index=True)
                        new_row_count += 1

                else:
                    print(f"Skipping {station_id_xx} - {evfile_name} - {phase_type} - {polarity}")
                    skipped_row_count += 1

        print(f"Added {new_row_count} new rows to the dataframe.")
        print(f"Skipped {skipped_row_count} rows.")

        # sort by file_name and then station_id
        pndting_df.sort_values(by=['file_name', 'station_id'], inplace=True)
        pndting_df.reset_index(drop=True, inplace=True)

        return pndting_df
    
    # ..............................................................................................
    #                   PYROCKO MARKER FILE TO CSV FILE
    # ..............................................................................................

    def PyrockoMarker2CSV(
        self, 
        pyrocko_markers: str,
        eq_cat_path: str,
        output_csv: str = None,
    ):
        """
        Convert Pyrocko marker file to a CSV file with columns:
        [event_id,etime,elat,elon,edep,emag,station_id,phase_type,phase_polarity,phase_time]
        N.B.: Same format as used for NCEDC picks to csv file.
        """
        # Read the EQ catalog file
        eq_cat_df = pd.read_csv(eq_cat_path, parse_dates=['time'])
        # Convert the pyrocko marker file to a dictionary
        data_dict = self.PyrockoMarker2Dict(pyrocko_markers)
        # Create an empty dataframe with the columns
        df = pd.DataFrame(
            columns=['event_id','etime','elat','elon','edep','emag','station_id','phase_type','phase_polarity','phase_time']
        )
        for event_id, phase_data in data_dict.items():
            # [station_id_xx, phase_time, phase_type, polarity]
            temp_df = pd.DataFrame(phase_data, columns=['station_id','phase_time','phase_type','phase_polarity'])
            temp_df['event_id'] = event_id
            temp_df['etime'] = eq_cat_df.loc[eq_cat_df.id == event_id, 'time'].values[0]
            temp_df['elat'] = eq_cat_df.loc[eq_cat_df.id == event_id, 'latitude'].values[0]
            temp_df['elon'] = eq_cat_df.loc[eq_cat_df.id == event_id, 'longitude'].values[0]
            temp_df['edep'] = eq_cat_df.loc[eq_cat_df.id == event_id, 'depth'].values[0]
            temp_df['emag'] = eq_cat_df.loc[eq_cat_df.id == event_id, 'mag'].values[0]
            # add the temp_df to the main dataframe
            df = pd.concat([df, temp_df], ignore_index=True)
        if output_csv is not None:
            df.to_csv(output_csv, index=False)
        return df
            



# ==================================================================================================
# SKHASH FILE FORMAT GENERATOR CLASS
# ==================================================================================================

class SkhashRunner:
    """
    Class convert PhaseNet picks file to SKHASH format.
    And to run the SKHASH program to generate focal mechanism solutions.
    """

    def __init__(self, *args, **kwargs):
        
        self.args = args
        self.kwargs = kwargs

        pass

    # ----------------------------------------------------------------------------------------------
    # SKHASH format polarity file generator
    # PhaseNet
    # ----------------------------------------------------------------------------------------------
    
    def PhaseNet2SKHASH_polarity(self,  
                eq_cat_path,
                PN_picks_path=None, 
                from_pyrocko_marker=False,
                manual_AI_commons_only=True, 
                pyrocko_only=True,
                events_in_cataglog_only=False, 
                output_path=None):
        """
        Convert the PhaseNet picks file to SKHASH format.
        event_id,event_id2,station,location,channel,p_polarity,origin_latitude,origin_longitude,origin_depth_km
        1,81905289,GTC,--,EHZ,0.02566,39.36743,-123.25380,7.14200

        Input: 
            PhaseNet picks dataframe/path to the file
            EQ catalog dataframe/path to the file

        Output: SKHASH polarity format dataframe
                - common events picked by both AI and manual
                    - if False: all diTing picks
                    - if True: only common picks
                - only manual picks (pyrocko picks)
                    - if false, use diting picks but common picks only
                    - if true, use only pyrocko picks
        """

        # Read the PhaseNet picks file
        # if given type is a dataframe then use it, otherwise read the file
        if not isinstance(PN_picks_path, pd.DataFrame):
            print("Reading the PhaseNet picks file")
            pol_df = pd.read_csv(PN_picks_path, parse_dates=["phase_time", "pyrocko_phase_time"])
        else:
            pol_df = PN_picks_path

        # Read the EQ catalog file
        if not isinstance(eq_cat_path, pd.DataFrame):
            eq_cat_df = pd.read_csv(eq_cat_path, parse_dates=['time'])
        else:
            eq_cat_df = eq_cat_path

        # create an empty dataframe with column names:
        skhash_pol_df = pd.DataFrame(
            columns=['event_id','event_id2','station','location','channel','p_polarity','origin_latitude','origin_longitude','origin_depth_km']
        )

        # Group by file_name
        i = 1
        grouped = pol_df.groupby('file_name')
        print(f"Total number of events: {len(grouped)}")

        for name, group in grouped:
            # Skip this group if manual_AI_commons_only is True and pyrocko picks are not available
            if manual_AI_commons_only and np.max(np.abs(group.pyrocko_polarity.astype(float))) != 1:
                continue
            
            # Get the event_id from the file_name (event_id.mseed)
            eventID = name.split('.')[0]

            # Run only on the events in the catalog?
            if events_in_cataglog_only:
                if eventID not in eq_cat_df.id.values:
                    continue
            
            # Get this group values into a skpol_df like empty dataframe
            event_df = pd.DataFrame(columns=skhash_pol_df.columns)
            
            # split the station_id into station, location, channel by '.'
            event_df.station = group.station_id.str.split('.').str[1]
            event_df.location = group.station_id.str.split('.').str[2].apply(lambda x: '--' if x == '' else x)
            
            if len(group.station_id[0].str.split('.').str[3]) == 2:
                event_df.channel = group.station_id.str.split('.').str[3] + 'Z'
            else:
                event_df.channel = group.station_id.str.split('.').str[3]
            

            # Fill the polarity column based on the pyrocko_only flag
            event_df.p_polarity = group.pyrocko_polarity    # add manual picks anyway
            if pyrocko_only:
                event_df.p_polarity = group.pyrocko_polarity
            else: 
                event_df.p_polarity = group.diting_polarity
                # drop if polarity = 'x'
                event_df = event_df[event_df.p_polarity != 'x']
                # convert the polarity to 1, -1, 0 from U, D, anything else
                # event_df.p_polarity = event_df.p_polarity.apply(lambda x: 1 if x == 'U' else -1 if x == 'D' else 0)
                # drop rows with polarity 0
                # event_df = event_df[event_df.p_polarity != 0]
            
            # VVI: Fill the event_id columns after the event_df is created with appropriate number of rows
            event_df.event_id = eventID
            event_df['event_id2'] = i
            i += 1

            # Get event details from EQcatalog
            eq_event = eq_cat_df[eq_cat_df.id == eventID].iloc[0] # get the first row if in case there are multiple rows
            event_df.origin_latitude = eq_event.latitude
            event_df.origin_longitude = eq_event.longitude
            event_df.origin_depth_km = eq_event.depth
            
            # Concatenate the event_df to skhash_pol_df if not empty
            if not event_df.empty:
                skhash_pol_df = pd.concat([skhash_pol_df, event_df], ignore_index=True)

        # drop rows if station/location/channel/polarity are empty
        skhash_pol_df = skhash_pol_df.dropna(subset=['station', 'location', 'channel', 'p_polarity'])

        if output_path:
            skhash_pol_df.to_csv(output_path, index=False)

        return skhash_pol_df
    # ----------------------------------------------------------------------------------------------
    # SKHASH format polarity file generator
    # 
    # NCEDC
    # ----------------------------------------------------------------------------------------------
    def NCEDC2SKHASH_polarity(
                self,
                eq_cat_path,
                picks_path,
    ):
        """
        Convert the NCEDC picks file to SKHASH format.
        """
        # create an empty dataframe with column names:
        skhash_pol_df = pd.DataFrame(
            columns=['event_id','event_id2','station','location','channel','p_polarity','origin_latitude','origin_longitude','origin_depth_km']
        )

        # Read the EQ catalog file
        eq_df = pd.read_csv(eq_cat_path,
                            names=['event_id', 'time', 'latitude', 'longitude', 'depth_km', 'mag'],
        )

        # Read the NCEDC picks file
        picks_df = pd.read_csv(picks_path,
                            names=['event_id', 'etime', 'elat', 'elon', 'edepth', 'mag', 'station_id', 'phase_type', 'phase_polarity', 'phase_time'],
        )
        # only keep the P picks
        picks_df = picks_df[picks_df.phase_type == 'P']
        # drop no polarity picks
        picks_df = picks_df.dropna(subset=['phase_polarity'])

        skhash_pol_df.event_id = picks_df.event_id
        skhash_pol_df.event_id2 = picks_df.event_id
        skhash_pol_df.station = picks_df.station_id.astype(str).str.split('.').str[1]
        skhash_pol_df.location = picks_df.station_id.astype(str).str.split('.').str[2].apply(lambda x: '--' if x == '' else x)
        skhash_pol_df.channel = picks_df.station_id.astype(str).str.split('.').str[3]
        skhash_pol_df.p_polarity = picks_df.phase_polarity.apply(lambda x: 1 if x == 'U' else -1 if x == 'D' else 0)
        skhash_pol_df.origin_latitude = picks_df.elat.round(5)
        skhash_pol_df.origin_longitude = picks_df.elon.round(5)
        skhash_pol_df.origin_depth_km = picks_df.edepth.round(3)

        return skhash_pol_df



    # ..............................................................................................
    #              Skhash station file generator function
    # ..............................................................................................

    def make_SKHASH_station_file(self,
        given_inventory = None,
        skhash_polarity_file = None,
        keep_Z_only = False,
        drop_duplicates = True,
        output_path = None,
        client_list = ['IRIS', 'NCEDC'], #'SCEDC'], 
        starttime = "2008-01-01", 
        endtime = "2023-01-01",
        region = [-125.5, -123, 39.75, 41.5],
        channel = 'HH*,BH*,HN*,EH*',
        buffer_zone: float = 2, # degrees
        ):

        """ 
        This function creates a station file for SKHASH format input data.
        No need to input a inventory file it will download the inventory from the client_list

        columns: 
        - `station,location,channel,latitude,longitude,elevation` 
        for this we will read the event_inventory.txt file.
        input:
            output_path: path to the output folder (mandatory)
            client_list: list of clients to download the inventory from (default: ['IRIS', 'NCEDC'])
            starttime: start time of the inventory (default: "2008-01-01")
            endtime: end time of the inventory (default: "2023-01-01")
            region: region of the inventory (default: [-128, -122.5, 39, 42])
            channel: channel of the inventory (default: 'HH*,BH*,HN*,EH*')
            given_inventory: path to the inventory file (default: None)/ master inventory file.txt
            output:
                station.csv file written to ouput folder
                or returns the dataframe
                
            """   
        
        if given_inventory == None:   
            # Either download station inventory for a large area,
            # or download the station inventory for 'all the stations' in the 
            # polarity file. 
            print(f"""
            Downloading the station inventory for
                            {region[3]}
                        _________________
                        |               |
                        |               |
            {region[0]} |               | {region[1]}
                        |               |
                        |_______________|
                            {region[2]}
            with a buffer zone of {buffer_zone} degrees.
            """)
            merged_inv = Inventory()
            for client_name in client_list:
                print(f"Downloading inventory from {client_name}")
                client = Client(client_name, debug=False, timeout=60)
                inv = client.get_stations(
                    starttime=starttime,
                    endtime=endtime,
                    minlongitude=region[0] - buffer_zone,
                    maxlongitude=region[1] + buffer_zone,
                    minlatitude=region[2] - buffer_zone,
                    maxlatitude=region[3] + buffer_zone,
                    channel="**Z,**3",
                    level="channel",
                )
                merged_inv += inv
            # Cleanup to remove duplicate stations
            merged_inv.write('./temp_inv.txt', format='stationtxt')
            invdf = pd.read_csv('./temp_inv.txt', sep='|', header=0
                ).drop_duplicates(subset=['#Network', 'Station', 'Location', 'Channel']
                ).to_csv('./temp_inv.txt', sep='|', index=False)
            given_inventory = './temp_inv.txt'

        else:
            print(f"Using the provided merged inventory file: {given_inventory}")

        # read the merged inventory file, keep only the required columns
        inv_df = pd.read_csv(given_inventory, sep='|', skiprows=1, 
                        usecols=[1, 2, 3, 4, 5, 6], # '#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime'
                        names=['station','location','channel','latitude','longitude','elevation'],
                        dtype={'station': str,'location': str, 'channel': str, 'latitude': float, 'longitude': float,
                                'elevation': float},    
                        )
        # fill the missing 'location' values with '--'
        inv_df.fillna({'location': '--'}, inplace=True)

        if keep_Z_only: # keep only the Z channel
            inv_df = inv_df[inv_df.channel.str[-1] == 'Z']

        if drop_duplicates:
            inv_df = inv_df.drop_duplicates(subset=['station', 'location', 'channel'])
        
        if skhash_polarity_file:
            skpol_df = pd.read_csv(skhash_polarity_file)
            # unique stations-location-channel
            unique_stations = skpol_df[['station', 'location', 'channel']].drop_duplicates()
            # keep only the stations that are in the polarity file
            inv_df = inv_df[inv_df.station.isin(unique_stations.station)]

        inv_df.sort_values(by=['station', 'location', 'channel'], inplace=True)
        if output_path:
            inv_df.to_csv(output_path, index=False)

        return inv_df

    # ..............................................................................................
    #              Skhash reverse file generator function
    # ..............................................................................................

    def make_skhash_reverse_file(self, skhash_polarity_file, output_path=None):
        """
        *** Make sure the polarity_file has been created in SKHASH format
        """

        df = pd.read_csv(skhash_polarity_file, header=0)
        
        reverse_df = pd.DataFrame()
        reverse_df['station'], reverse_df['location'], reverse_df['channel'] = df['station'], df['location'], df['channel']
        reverse_df.drop_duplicates(subset=['station'], inplace=True)

        # put dummy start and end time
        reverse_df['start_time'] = "1900-01-01"
        reverse_df['end_time'] = "2200-01-01"

        if output_path:
            reverse_df.to_csv(output_path, index=False)

        return reverse_df

    # ..............................................................................................
    #              Skhash control file editor function
    # ..............................................................................................

    def edit_skhash_control_file(self, control_file = None, output_path=None,
                                conpfile = 'examples/maacama_SKHASH_MTJ/IN/pol_consensus.csv',
                                stfile = 'examples/maacama_SKHASH_MTJ/IN/station.csv',
                                outfile1 = 'examples/maacama_SKHASH_MTJ/OUT/out.csv',
                                outfile2 = 'examples/maacama_SKHASH_MTJ/OUT/out2.csv',
                                outfile_pol_agree = 'examples/maacama_SKHASH_MTJ/OUT/out_polagree.csv',
                                outfile_pol_info = 'examples/maacama_SKHASH_MTJ/OUT/out_polinfo.csv',
                                vmodel_paths = 'examples/velocity_models_MTJ/vz.MTJ.txt',
                                output_angle_precision = 4,
                                require_network_match : bool = False,
                                allow_duplicate_stations : bool = True,
                                min_polarity_weight = 0,
                                dang = 5,
                                nmc = 30,
                                maxout = 500, 
                                ratmin = 2,
                                badfrac = 0.1,
                                qbadfrac = 0.3,
                                delmax = 175, # maximum allowed source-receiver distance in km.
                                max_pgap = 65,
                                cangle = 45,
                                prob_max = 0.2,
                                azmax = 0,
                                max_agap = 135, # in MTJ there's usually no stations in the offshore area
                                outfolder_plots = None,
                                plot_station_names = False,
                                plot_acceptable_solutions = False,

                                ):
        """
        Edit the SKHASH control file to include the paths to the station, polarity, and reverse files.
        """

        orig_file = f"""## Control file example for Maacama (SKHASH/examples/maacama_SKHASH_MTJ)

$input_format  # format of input files
SKHASH

$num_cpus      # number of CPUs to use
8

$conpfile        # P-polarity input filepath
{conpfile}

$stfile        # station list filepath
{stfile}

$outfile1      # focal mechanisms output filepath 
{outfile1}

$outfile2 : # Path to acceptable plane output file
{outfile2}

$outfile_pol_agree  # record of polarity (dis)agreeement output filepath # examples/maacama_SKHASH_MTJ/OUT/out_polagree.csv
{outfile_pol_agree}

$outfile_pol_info # examples/maacama_SKHASH_MTJ/OUT/out_polinfo.csv
{outfile_pol_info}

$vmodel_paths  # whitespace/newline delimited list of paths to the velocity models 
{vmodel_paths}

$output_angle_precision
{output_angle_precision}

$require_network_match
{require_network_match}

$allow_duplicate_stations
{allow_duplicate_stations}

$min_polarity_weight
{min_polarity_weight}

$dang          # minimum grid spacing (degrees)
{dang}

$nmc           # number of trials (e.g., 30)
{nmc}

$maxout        # max num of acceptable focal mech. outputs (e.g., 500)
{maxout}

$ratmin        # minimum allowed signal to noise ratio
{ratmin}

$badfrac       # fraction polarities assumed bad
{badfrac}

$qbadfrac      # assumed noise in amplitude ratios, log10 (e.g. 0.3 for a factor of 2)
{qbadfrac}

$delmax        # maximum allowed source-receiver distance in km.
{delmax}

$max_pgap      # maximum allowed takeoff angle in degrees
{max_pgap}

$cangle        # angle for computing mechanisms probability
{cangle}

$prob_max      # probability threshold for multiples (e.g., 0.1)
{prob_max}

$azmax         # Maximum allowed source-station azimuth uncertainty in degrees [0 = all allowed]
{azmax}

$max_agap      # maximum azimuthal gap in degrees
{max_agap}

$allow_hypocenters_outside_table # False: only hypocenters within the velocity model are allowed
True

"""
        
        if outfolder_plots is not None:
            orig_file += f"""\n$outfolder_plots : #Path to folder where simple focal mechanism plots \n{outfolder_plots}\n"""
        if plot_station_names:
            orig_file += f"""\n$plot_station_names : #Plot station names \n{plot_station_names}\n"""
        if plot_acceptable_solutions:
            orig_file += f"""\n$plot_acceptable_solutions : #Plot acceptable solutions \n{plot_acceptable_solutions}\n"""

        return orig_file
        

# ==================================================================================================
#                                        
# ==================================================================================================