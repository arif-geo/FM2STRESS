import os
import numpy as np
import pandas as pd
from obspy import read, UTCDateTime, Stream, Inventory
import matplotlib.pyplot as plt



def WFprocessing(
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
    temp_st = st.select(id=stn_id)
    print(stn_id)
    tr = temp_st[0].copy()

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


def get_trace2plot(row, st, slice_len, normalize, phasetime_col, 
                polarity_col='diting_polarity',
                    ):
    """
    Prepare the trace for plotting with PhasesNet and Pyrocko picks and polarities.
        To be used within the plotter functions. 
    """

    # get the pick times (PhasesNet and Pyrocko)
    pn_pick = UTCDateTime(pd.to_datetime(str(row[phasetime_col])))

    # get the trace for this station
    stn_id = f'{row.station_id}Z'

    # Process the trace using the WFprocessing function
    tr = WFprocessing(stn_id, pn_pick, st, slice_len, normalize)
    
    if len(tr.data) == 0:
        return None, None, None, None

    # Generate time axis as a numpy array
    times = np.linspace(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.npts) # time [s]

    # get the pick time from starttime into seconds
    pn_pick_time = ((pn_pick - tr.stats.starttime)) 
    
    # get polarity
    dting_pol = row[polarity_col]
    
    dting_sharp = row['diting_sharpness']

    return tr, times, pn_pick_time, dting_pol


def plotWFpick_pol(
                    mseed_file = None,
                    pol_df = None,
                    waveform_dir = None,
                    phasetime_col = 'phase_time',
                    figsize=(10, 6),
                    n_subplots=10, 
                    slice_len=0.5,
                    zoom=5,
                    normalize: bool = True,
                    hor_line: bool = True,
                    
                    ):

    # subset the df for this event only
    event_df = pol_df[pol_df['file_name'] == mseed_file].dropna(subset=[phasetime_col]).reset_index(drop=True)

    # read the mseed file
    st = read(f'{waveform_dir}/{mseed_file}')
    st.resample(100)            # resample to 100 Hz

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if n_subplots == 'all' or n_subplots > len(event_df):
        n_subplots = len(event_df)


    for i, row in event_df[0:n_subplots].iterrows():
        
        # prepare the trace for plotting with PhasesNet and Pyrocko picks and polarities
        tr, times, pn_pick_time, dting_pol = get_trace2plot(
            row, st, slice_len, normalize, phasetime_col
        )

        if tr is None: # skip if the trace is empty
            print(f"Skipping {row.station_id}Z")
            continue

        # variable zoom using spread of the data
        spread = np.std(tr.data)          
        vzoom = zoom/np.round(spread, 2)/10 if spread != 0 else zoom

        ax.plot(times, tr.data * vzoom + i, 'k', lw=0.5)     # plot the waveform
        ax.plot([pn_pick_time, pn_pick_time], [i-0.4, i+0.4], color='r', ls='--')      # PhasesNet pick
        # Polarity and sharpness (DiTing)
        ax.text(
            pn_pick_time-0.15, i + 0.5, 
            f"{dting_pol}", 
            fontsize=12, color='b', ha='left')

        # title [station_id] and horizontal line at 0
        ax.text( 
            pn_pick_time-0.5,
            i - 0.3, 
            f"{tr.id}",
            fontsize=8)

    # # add one title for each column
    ax.set_title('PhasesNet and DiTingMotion', fontsize=14)

    
    # plot axes (one for both columns)
    ax.set_ylabel('Amplitude', fontsize=12)
    plt.xlabel('Time [s]',x=-.05, y=0.01, fontsize=12)

    return fig, ax

# --------------------------------------------------------------------------------------------
def plotWFpick_pol2(mseed_file = None,
                    pol_df = None,
                    waveform_dir = None,
                    phasetime_col = 'phase_time',
                    polarity_col='diting_polarity',
                    figsize=(10, 20),
                    n_subplots=10, 
                    slice_len=0.5,
                    zoom=5,
                    normalize: bool = True,
                    hor_line: bool = True,
                    
                    ):

    # subset the df for this event only
    event_df = pol_df[pol_df['file_name'] == mseed_file]#.dropna(subset=[phasetime_col]).reset_index(drop=True)

    # read the mseed file
    st_o = read(f'{waveform_dir}/{mseed_file}')
    st = st_o.copy()
    st.resample(100)            # resample to 100 Hz

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if n_subplots == 'all' or n_subplots > len(event_df):
        n_subplots = len(event_df)

    for i, row in event_df[0:n_subplots+1].iterrows():
        
        # prepare the trace for plotting with PhasesNet and Pyrocko picks and polarities
        
        # get the pick times (PhasesNet and Pyrocko)
        pn_pick = UTCDateTime(pd.to_datetime(str(row[phasetime_col])))

        # get the trace for this station
        stn_id = f'{row.station_id}Z'
        # print(stn_id, pn_pick)

        # Process the trace
        tr = st.select(id=stn_id)[0]

        # check for empty trace
        if len(tr.data) > 0:
            tr.detrend('demean')
            try:
                tr.detrend('linear')
            except:
                tr.detrend('constant')
            
            try:
                tr.taper(0.001)
                tr.filter('bandpass', freqmin=1.5, freqmax=10, corners=4, zerophase=True)  # Apply a bandpass filter
            except:
                pass
            
            if normalize == True:
                tr.data = tr.data/np.max(tr.data) # normalize the trace

        
        tr = tr.slice(pn_pick - (slice_len-0.01), pn_pick + slice_len) # around PhasesNet pick

        if len(tr.data) == 0:
            continue


        # Generate time axis as a numpy array
        times = np.linspace(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.npts) # time [s]

        # get the pick time from starttime into seconds
        pn_pick_time = ((pn_pick - tr.stats.starttime)) 
        
        # get polarity
        dting_pol = row[polarity_col]

        # variable zoom using spread of the data
        spread = np.std(tr.data)          
        vzoom = zoom/np.round(spread, 2)/10 if spread != 0 else zoom
        
        # plot the waveform
        ax.plot(times, tr.data * vzoom + i, 'k', lw=0.5)     # plot the waveform

        # plot the arrival time pick
        ax.plot([pn_pick_time, pn_pick_time], [i-0.4, i+0.4], color='r', ls='--')      # PhasesNet pick
        
        # Polarity and sharpness (DiTing)
        ax.text(
            pn_pick_time-0.15, i + 0.5, 
            f"{dting_pol}", 
            fontsize=12, color='b', ha='left')

        # title [station_id] and horizontal line at 0
        ax.text( 
            pn_pick_time-0.5,
            i - 0.3, 
            f"{tr.id}",
            fontsize=8)

    # # add one title for each column
    ax.set_title('PhasesNet and DiTingMotion', fontsize=14)

    
    # plot axes (one for both columns)
    ax.set_ylabel('Amplitude', fontsize=12)
    plt.xlabel('Time [s]',x=-.05, y=0.01, fontsize=12)

    return fig, ax



def plot_wf_check_picks(
    pn_pick_df,
    wf_dir_processed,
    mseed_list_processed,
    selected_file,
):
    event_df = pn_pick_df[pn_pick_df["file_name"] == selected_file]
    st = read(f"{wf_dir_processed}/{selected_file}")

    fig, axs = plt.subplots(len(event_df), 1, figsize=(10, 1 * len(event_df)))
    axs = axs.flatten()

    iplt = 0
    for i, row in event_df.iterrows():
        ist = st.select(id=f"{row.station_id}Z")
        if len(ist) == 0:
            continue
        tr = ist[0].copy()
        tr.detrend('demean')
        try:
            tr.detrend('linear')
        except:
            tr.detrend('constant')
        
        try:
            tr.taper(0.001)
            tr.filter('bandpass', freqmin=1.0, freqmax=20)
        except:
            pass

        # tr.data = tr.data/np.max(tr.data) # normalize the trace

        phase_time = UTCDateTime(pd.to_datetime(str(row.phase_time)))
        dt_polarity = row.diting_polarity

        if len(tr) == 0:
            continue
        # if phase_time > tr.stats.endtime:
        #     continue

        xtimes = np.linspace(0, len(tr.data)/tr.stats.sampling_rate, len(tr.data))
        pick_sec = np.round(phase_time - tr.stats.starttime, 2)

        # print(tr.id, xtimes[-1], pick_sec)

        axs[iplt].plot(xtimes, tr.data, color='k')              # plot the waveform
        axs[iplt].axvline(pick_sec, color='r', linestyle='--')  # plot the pick time
        axs[iplt].text(0, 0.25, f"{tr.id} - {row.phase_score} - {row.sta_dist_km}",)                   # set the station id
        axs[iplt].text(pick_sec-0.1, 0.2, dt_polarity, color='r')     # set the polarity

        # hide x labels and ticks
        axs[iplt].get_xaxis().set_visible(False) 

        iplt += 1

    plt.suptitle(f"{selected_file.split('.')[0]}")
    plt.tight_layout()
    plt.show()
    return fig, axs