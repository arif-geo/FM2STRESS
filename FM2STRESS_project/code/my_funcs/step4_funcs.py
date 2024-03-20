"""
I/O operation for HASH
"""
import pandas as pd
import glob
import numpy as np
import pygmt

def read_output1(filename, amp=False):
    """
    event type (L-local, R-regional, T -teleseism Q-quarry, D-dubious or duplicate event)
    mag type e energy magnitude
            w moment magnitude
            b body-wave magnitude
            s surface-wave magnitude
            l local (Wood-Anderson)
            c coda amplitude magnitude 
            h heliocorder magnitude
            d coda duration magnitude
    location quality (SCEC qualities)
            A rms<0.15,erh<1 kmerz<2km 
            B rms< 0.30, erh< 2.5 km erz< 5 km 
            C rms< 0.50, erh< 5 km ---
            D greater than above
            E erh> 90 km or < 3 stations
    eRMS, RMS from SCEC or L-2 norm type locations
    err1, fault plane uncertainty (degrees)
    err2, auxiliary plane uncertainty (degrees)
    weight, weighted percent misfit of first motions
    prob, probability mechanism close to solution 
    st_ratio, 100*(station distribution ratio)
    """
    if amp == False:
    
        df = pd.read_csv(filename, header=None,
                    sep="\s+",
                    skipinitialspace=True, index_col=False,
                     names=['event', 
                            'year', 'month', 'day',
                            'hour', 'minute', 'sec', 
                            'type', 'mag', 'mmagtype',
                            'elat', 'elon', 'edep', 'equal', 'eRMS', 
                            'hori_err', 'dep_err', 'ori_time_err',
                            'n_pick', 'n_p_pick', 'n_s_pick', 
                            'strike', 'dip', 'rake', 'err1', 'err2', 
                            'n_pol', 'weight', 'f_qual', 'prob', 'st_ratio', '*'],
                      dtype={'event':str, 'year':int, 'month':int, 'day':int, 'hour':int,
                            'minute':int, 'sec':float, 'type':str, 'mag':float, 'mmagtype':str,
                            'elat':float, 'elon':float, 'edep':float, 'e_qual':str, 'eRMS':float, 
                            'hori_err':float, 'dep_err':float, 'ori_time_err':float,
                            'n_pick':int, 'n_p_pick':int, 'n_s_pick':int, 
                            'strike':int, 'dip':int, 'rake':int, 'err1':int, 'err2':int, 
                            'n_pol':int, 'weight':int, 'f_qual':str, 'prob':int, 'st_ratio':int})
    return df


#==================================================================================================

def get_polarity(filename_polarity, folder_station, items):
    """
    :param filename_polarity: polarity file
    :param folder_station: folder of station files
    :param items: items to be extracted from station files (see get_sta_dict)

    :return: 
            polarities: array of polarities, [polarity, slat, slon]
            stanames: list of station names
    """
    from step2_funcs import get_sta_dict
    sta_loc = get_sta_dict(folder_station, items)

    df = pd.read_csv(filename_polarity, sep='\s+', header=None, skiprows=1, usecols=[4,8,9], \
                     names=['sta', 'phase', 'polarity'],\
                     dtype={'sta':str,'phase':str, 'polarity': str})
    polarities = []
    stanames = []
    for i in range(0, df.shape[0]):
        stastr, phasetype, polarity = df['sta'][i], df['phase'][i], df['polarity'][i]
        if polarity == '1' and phasetype == 'P':
            sta = stastr.split('.')[1]
            slat, slon = sta_loc[sta][0], sta_loc[sta][1]
            polarities.append([1, slat, slon])
            stanames.append(sta)

        elif polarity == '-1' and phasetype == 'P':
            sta = stastr.split('.')[1]
            slat, slon = sta_loc[sta][0], sta_loc[sta][1]
            polarities.append([-1, slat, slon])
            stanames.append(sta)
    polarities = np.array(polarities)
    return polarities, stanames
#==================================================================================================

def get_polarity2(event_folder, event_id):
    """ 
    .phase file created earlier will be used to get polarity and station names that are used to pick phases and polarity.
    don't read line 1 and last line.
    col names [station , net, channel, I, poarity(U/D)]
    """
    phase_file = f"{event_folder}/{event_id}.phase"
    df = pd.read_csv(phase_file, sep='\s+', header=None, skiprows=1, usecols=[0,4],
                    names=['station', 'polarity'], dtype={'station':str, 'polarity':str})
    # drop the last row 
    df = df[:-1]

    df2 = get_sta_latlon(event_folder)
    print(df.shape, df2.shape)

    # Merge df1 and df2 on the station name, keeping all rows from df1
    df = df.merge(df2[['Station', 'Latitude', 'Longitude']], how='left', left_on='station', right_on='Station')

    # Drop the extra 'Station_all' column
    df = df.drop('Station', axis=1)
    # Rename columns for clarity
    df = df.rename(columns={'station': 'Station', 'polarity': 'Polarity'})  # Optional

    return df

#==================================================================================================

def get_sta_latlon(event_folder):
    """
    :param event_folder: folder of station files .txt
        folder structure: data/{event_id}/files.foramat

    :return:
            stanames: list of station names
            latlon: array of station latitudes and longitudes
    """
    sta_file = f"{event_folder}/event_inventory.txt"
    
    df = pd.read_csv(sta_file, 
                    sep='|', 
                    header=0,
                    usecols=[1,4,5]
                    )
    # drop duplicate stations
    df = df.drop_duplicates(subset=['Station'])
    df = df.reset_index(drop=True)
    
    return df

#==================================================================================================

def plot_focal(region, strike, dip, rake, elat, elon, edep, magnitude: float):
    """
    :param region: region to plot [west, east, south, north]
    :param strike, dip, rake: focal mechanism parameters
    :param elat, elon, edep: event location

    :return: pygmt figure
    """
    fig = pygmt.Figure()
    # generate a basemap near Washington state showing coastlines, land, and water
    pygmt.config(MAP_ANNOT_OBLIQUE="lat_parallel")

    fig.coast(
        region=region,
        projection="M12c",
        land="grey",
        water="lightblue",
        shorelines=True,
        resolution="f",
        frame="a",
    )
    
    fig.basemap(
        region=region,
        projection="M12c",
        frame=["WSne", "xaf+lLongitude", "yaf+lLatitude"]
        )
    # store focal mechanisms parameters in a dict
    focal_mechanism = dict(strike=strike, dip=dip, rake=rake, magnitude=magnitude)

    # pass the focal mechanism data to meca in addition to the scale and event location
    fig.meca(focal_mechanism, scale="1c", longitude=elon, latitude=elat, depth=edep)
    return fig