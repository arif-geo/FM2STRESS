from scipy.interpolate import griddata
import pandas as pd
from obspy import UTCDateTime, read
import numpy as np
import matplotlib.pyplot as plt
import utm
import pdb
import glob
from geopy.distance import geodesic
import os


def convert_lat_lon_to_xy(latitude, longitude, radius=6371):
    """
    Convert latitude and longitude to x, y
    Input:
        latitude: np.array or float
        longitude: np.array or float
    Output:
        x: np.array or int, km
        y: np.array or int, km
    """
    # Convert latitude and longitude from degrees to radians
    # lat_rad = np.radians(latitude)
    # long_rad = np.radians(longitude)

    # use utm module to convert lat/lon to x/y
    x, y, _, _ = utm.from_latlon(latitude, longitude)
    x, y = x/1000, y/1000  # convert to km 

    return x, y

#==================================================================================================

def get_traveltime_points_value(filename):
    """
    Input:
        filename: filename of the P or S traveltime table
    Output:
        points and values required by the scipy.interpolate.griddata function
        points: np.array(N_grid, 2)
            1st column distance, 2nd column dep
        value: np.array(N_grid, )
            corresponding traveltime from the traveltime table
    Please check the lecture notes!!!
    """
    ########### Ginny's version ############
    # dep = np.arange(0, 80.1, 0.5)
    # dist = np.arange(0, 150.1, 0.5)
    # df = pd.read_csv(filename, skiprows=3, sep='\s+', header=None)
    # matrix = df.values
    # ndist, ndep = df.shape[0], df.shape[1] - 1
    # points = np.zeros((ndist * ndep, 2))
    # values = np.zeros((ndist * ndep,))
    # count = 0
    # for i in range(0, ndist):
    #     for j in range(0, ndep):
    #         points[count, :] = [dist[i], dep[j]]
    #         values[count] = matrix[i, j + 1]
    #         count += 1
    # return points, values

    ########### Arif's version ############
    # read the file
    df_raw = pd.read_csv(filename, skiprows=2, header=None)

    df = pd.read_csv(filename, skiprows=3,
                    header=None, sep='\s+')

    # distance in first column
    distances = df.iloc[:, 0].values
    # depth in 3rd row of the header / 1st column of the df_raw 
    depths = df_raw.iloc[0, :] # returns a bad format series
    depths = pd.DataFrame(depths)[0][0] # convert to string
    depths = np.array(depths.split()).astype(float) # convert to array of floats

    # points = np.zeros((len(distances), 2))
    points = []
    values = []
    for i, dist in enumerate(distances):
        for j, dep in enumerate(depths):
            points.append([dist, dep])
            values.append(df.at[i, j+1]) # j+1 because the first column is distance, it will 
                                         # return distance for j=0, not the first depth value

    # convert to np.array
    points = np.array(points)
    values = np.array(values)

    return points, values

#==================================================================================================

def get_misfit(array1, array2):
    """
    Input:
        array1: np.array or float
        array2: np.array or float
        Here, array1 is the demeaned predicted traveltime,
              array2 is the demeaned arrival time
    Output:
        misfit: np.array or float
    Calculate norm 1 or norm 2 misfit between array1 and array2

    """
    misfit = np.sum(np.abs(array1-array2))
    return misfit

#==================================================================================================

def get_mean_arrival(arrival_time):
    """
    Input:
        arrival_time: list
    """
    n = len(arrival_time)
    min_arrival_time = np.min(arrival_time)
    dt = np.zeros((n, ))
    for i in range(0, n):
        dt[i] = arrival_time[i] - min_arrival_time
    return min_arrival_time + np.mean(dt)

#==================================================================================================

def get_phasepicks_list(filename):
    # [phasetime]
    df = pd.read_csv(filename, sep='\s+', header=None, skiprows=1, usecols=[1,2,4,8], \
                     names=['date','time','sta','phase'],\
                     dtype={'date':str,'time':str,'sta':str,'phase':str})
    phase_list = {}
    for i in range(0, df.shape[0]):
        dd, tt, stastr, phasetype = df['date'][i], df['time'][i], df['sta'][i], df['phase'][i]
        phasetime = UTCDateTime('{}T{}'.format(dd, tt))
        sta = stastr.split('.')[1]
        phase_list['{}-{}'.format(sta, phasetype)] = phasetime
    return phase_list

#==================================================================================================

def get_polarity(filename, folder_station):
    """
    Input:
        filename: filename of the polarity file
        folder_station: folder containing station information
    Output: 
        polarities: np.array((npick, 3))
            3 columns are: polarity (1 for positive, -1 for negative),
                           station latitude,
                           station longitude
        stanames: list
            list of station names
    """

    sta_loc = get_sta_loc(folder_station)

    df = pd.read_csv(filename, sep='\s+', header=None, skiprows=1, usecols=[4,8,9], \
                     names=['sta', 'phase', 'polarity'],\
                     dtype={'sta':str,'phase':str, 'polarity': str})
    polarities = []
    stanames = []
    for i in range(0, df.shape[0]):
        stastr, phasetype, polarity = df['sta'][i], df['phase'][i], df['polarity'][i]
        if polarity == '1' and phasetype == 'P':
            sta = stastr.split('.')[1]
            slon, slat = sta_loc[sta][0], sta_loc[sta][1]
            polarities.append([1, slat, slon])
            stanames.append(sta)
        elif polarity == '-1' and phasetype == 'P':
            sta = stastr.split('.')[1]
            slon, slat = sta_loc[sta][0], sta_loc[sta][1]
            polarities.append([-1, slat, slon])
            stanames.append(sta)
    polarities = np.array(polarities)
    return polarities, stanames

#==================================================================================================

def read_phase_picks(filename_picks, folder_station):
    """
    Input:
        filename_picks: filename of the phasepicks
        filename_station: filename of the station information
    Output:
        phasepicks: np.array((npicks, 4))
            4 columns are: station x position,
                           station y position,
                           phasetype (1 for P, 2 for S),
                           demeaned arrival time
        mean_arrival_time: UTCDateTime format
            mean of all the phase pick times
    """
    sta_file = f"{folder_station}/event_inventory.txt"

    # read station inventory file
    sta_df = pd.read_csv(sta_file, 
                        sep='|', 
                        header=0,
                        usecols=[1, 4, 5]) # Station|Latitude|Longitude    
    # keep only unique station names
    sta_df = sta_df.drop_duplicates(subset=['Station'])

    # read phase picks file
    df_picks = pd.read_csv(filename_picks, sep='\s+', header=None, skiprows=1, usecols=[1, 2, 4, 8],
                     names=['date', 'time', 'sta', 'phase'],
                     dtype={'date': str, 'time': str, 'sta': str, 'phase': str})
    
    # get number of picks
    npick = df_picks.shape[0]

    # get station name, latitude, longitude
    staname, slat, slon = sta_df['Station'].values, sta_df['Latitude'].values, sta_df['Longitude'].values
    
    # convert lat, lon to x, y
    sx, sy = convert_lat_lon_to_xy(slat, slon)

    # Create an array of shape (len(sta_df), 2) to hold [slon, slat] pairs
    sta_xy = np.column_stack((sx, sy))

    # Create sta_loc dictionary using broadcasting
    sta_loc = dict(zip(staname, sta_xy))
    
    

    # define an empty np.array to store phase picks 
    phasepicks = np.zeros((npick, 4))

    # loop over all picks
    arrival_time = []
    for i in range(0, npick):
        # get station name
        stastr = df_picks['sta'][i]
        staname = stastr.split('.')[1]

        # get station x, y, and phasetype
        sx, sy = sta_loc[staname]

        if df_picks['phase'][i] == 'P':
            phasetype = 1
        else:
            phasetype = 2

        arrival_time.append(UTCDateTime('{}T{}'.format(df_picks['date'][i], df_picks['time'][i])))
        
        # append x, y and phasetype to phasepicks
        phasepicks[i, 0:3] = [sx, sy, phasetype]

    mean_arrival_time = get_mean_arrival(arrival_time)
    for i in range(0, npick):
        # append demeaned arrival time to phasepicks
        phasepicks[i, 3] = arrival_time[i] - mean_arrival_time
    return phasepicks, mean_arrival_time

#==================================================================================================

def get_sta_lat_lon(filename_picks, folder_station):
    """
    Input:
        filename_picks: filename of the phasepicks
        folder_station: folder containing station information
    Output:
        lat and lon of all the stations that are in the phasepicks file i.e. the stations 
        that has been used to pick the event(s)

    N.B. almost same as the following function get_sta_loc
    """
    sta_file = f"{folder_station}/event_inventory.txt"

    # read station inventory file
    sta_df = pd.read_csv(sta_file, 
                        sep='|', 
                        header=0,
                        usecols=[1, 4, 5]) # Station|Latitude|Longitude    
    # keep only unique station names
    sta_df = sta_df.drop_duplicates(subset=['Station'])

    # Create sta_loc dictionary using broadcasting
    sta_loc = dict(zip(sta_df.Station.values, zip(sta_df.Longitude.values, sta_df.Latitude.values)))

    # read phase picks file
    df_picks = pd.read_csv(filename_picks, sep='\s+', header=None, skiprows=1, usecols=[1, 2, 4, 8],
                     names=['date', 'time', 'sta', 'phase'],
                     dtype={'date': str, 'time': str, 'sta': str, 'phase': str})
    npick = df_picks.shape[0]

    sta_position = np.zeros((npick, 2))
    for i in range(0, npick):
        stastr = df_picks['sta'][i]
        staname = stastr.split('.')[1]
        slon, slat = sta_loc[staname]
        sta_position[i, :] = [slon, slat]
    return sta_position[:,0], sta_position[:,1] # lon, lat

#==================================================================================================
def get_sta_loc(folder_station):

    """
    Input:
        folder_station: folder containing station information
    Output: 
        sta_loc: dictionary
            key: station name
            value: [longitude, latitude]
    """

    sta_file = f"{folder_station}/event_inventory.txt"

    # read station inventory file
    sta_df = pd.read_csv(sta_file, 
                        sep='|', 
                        header=0,
                        usecols=[1, 4, 5]) # Station|Latitude|Longitude    
    # keep only unique station names
    sta_df = sta_df.drop_duplicates(subset=['Station'])

    # Create sta_loc dictionary using broadcasting
    sta_loc = dict(zip(sta_df.Station.values, zip(sta_df.Longitude.values, sta_df.Latitude.values)))

    return sta_loc

#==================================================================================================
def get_sta_dist(folder_station, elat, elon):
    """
    For plotting
    folder_station: folder containing station information (such as ../data/eq_data/event_id/)
    """
    sta_file = f"{folder_station}/event_inventory.txt"

    # read station inventory file
    sta_df = pd.read_csv(sta_file, 
                        sep='|', 
                        header=0,
                        usecols=[1, 4, 5]) # Station|Latitude|Longitude    
    # keep only unique station names
    sta_df = sta_df.drop_duplicates(subset=['Station'])

    stanames, slat, slon = sta_df['Station'].values, sta_df['Latitude'].values, sta_df['Longitude'].values
    distances = []
    for i in range(0, len(sta_df)):
        distances.append(geodesic((elat, elon), (slat[i], slon[i])).km)
    distances = np.array(distances)
    return stanames, distances

#==================================================================================================
    
