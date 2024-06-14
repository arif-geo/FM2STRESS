"""
I/O operation for HASH
"""
import pandas as pd
import glob
import numpy as np

def make_eventfile(event_id, eqfile, outfile):
    """
    convert eqfile to HASH event file
    output: 
           year month day hour minute second latitude longitude depth hori_err vert_err mag    EVID
    """
    df = pd.read_csv(eqfile, sep=',', header=0, parse_dates=['time'])
    df = df[df['id'] == event_id]
    df = df.reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])
    
    # create a blank df with column names 
    df2 = pd.DataFrame(columns=['year', 'month', 'day', 'hour', 'minute', 'second', 'latitude', 'longitude', 'depth', 'hori_err', 'vert_err', 'mag', 'EVID'])
    
    df2['year'], df2['month'], df2['day'] = df['time'].dt.year, df['time'].dt.month, df['time'].dt.day
    df2['hour'], df2['minute'], df2['second'] = df['time'].dt.hour, df['time'].dt.minute, df['time'].dt.second
    df2['latitude'], df2['longitude'], df2['depth'] = df['latitude'], df['longitude'], df['depth']
    df2['hori_err'], df2['vert_err'] = df['horizontalError'], df['depthError']
    df2['mag'] = df['mag']
    df2['EVID'] = event_id

    df2.to_csv(outfile, sep=',', header=True, index=False)


def snuffler_to_HASH(filename_event, folder_station, filename_picks, outfile):
    """
    convert snuffler phase pick file to HASH input, drive 2 format
    old SCEDC phase format
    station names need 5 letters
    """
    # read event
    df_event = pd.read_csv(filename_event, header=0)
    eyear, emonth, eday, ehour, eminute, esec = df_event['year'][0], df_event['month'][0], df_event['day'][0], \
                                            df_event['hour'][0], df_event['minute'][0], df_event['second'][0]    
    
    elat_int = int(np.abs(df_event['latitude'].iloc[0]))
    if df_event['latitude'][0] < 0:
        elat_char = 'S'
        elat_minute = (-df_event['latitude'][0]-elat_int)*60
    else:
        elat_char = ' '
        elat_minute = (df_event['latitude'][0]-elat_int)*60
    
    elon_int = int(np.abs(df_event['longitude'].iloc[0]))
    if df_event['longitude'][0] > 0:
        elon_char = 'E'
        elon_minute = (df_event['longitude'][0]-elon_int)*60
    else:
        elon_char = ' '
        elon_minute = (-df_event['longitude'][0]-elon_int)*60
    
    # event depth, horizontal and vertical errors, magnitude, event ID
    edep = df_event['depth'][0]
    e_hori_err, e_vert_err = df_event['hori_err'][0], df_event['vert_err'][0]
    emag, eventID = df_event['mag'][0], df_event['EVID'][0]
    
    # read station
    # sta_loc = get_sta_dict(folder_station, ['Latitude','Longitude'])
    
    # read phase picks
    df_picks = pd.read_csv(filename_picks, sep='\s+', header=None, skiprows=1, usecols=[1, 2, 4, 8, 9],
                     names=['date', 'time', 'sta', 'phase', 'pol'],
                     dtype={'date': str, 'time': str, 'sta': str, 'phase': str, 'pol': str})
    
    # drop rows if 'phase' is not 'P'
    df_picks = df_picks[df_picks['phase'] == 'P'].reset_index(drop=True)

    npick = df_picks.shape[0]
    print('npick', npick)
    
    # empty chars
    str50 = ''
    for i in range(0, 49):
        str50 = str50 + ' '
    str41 = ''
    for i in range(0, 40):
        str41 = str41 + ' '
    str7 = ''
    for i in range(0, 6):
        str7 = str7 + ' '
    str56 = ''
    for i in range(0, 56):
        str56 = str56 + ' '
    
        
    fid = open(outfile, 'w')
    # Event line
    line = '{:04d}{:>2d}{:>2d}{:>2d}{:>2d}{:>5.2f}{:>2d}{}{:>5.2f}{:>3d}{}{:>5.2f}{:>5.2f}{}{:>5.2f} {:>5.2f}{}{:>4.2f}{}{:>16}\n'.format(
                    eyear, emonth, eday, ehour, eminute, esec, 
                    elat_int, elat_char, elat_minute, elon_int, elon_char, elon_minute, edep,
                    str50, e_hori_err, e_vert_err, str41, emag, str7, eventID                    
    )

    fid.write(line)

    stas_pick_dict = {}
    # Polarity lines
    for i in range(0, npick):

        staname = df_picks['sta'][i].split('.')[1]
        network = df_picks['sta'][i].split('.')[0]
        comp = df_picks['sta'][i].split('.')[-1]
        polstr = df_picks['pol'][i]
        if polstr == '1':
            if staname in stas_pick_dict.keys():
                if polstr == stas_pick_dict[staname]:
                    continue
                elif polstr != stas_pick_dict[staname]:
                    print('polarity opposition on the same station {}.{}'.format(network, staname))
                    continue
            pol = 'U'
            fid.write('{:<5} {}  {} I U\n'.format(staname, network, comp))
            stas_pick_dict[staname] = polstr

        elif polstr == '-1':
            if staname in stas_pick_dict.keys():
                if polstr == stas_pick_dict[staname]:
                    continue
                elif polstr != stas_pick_dict[staname]:
                    print('polarity opposition on the same station {}'.format(staname))
                    continue
            pol = 'D'
            fid.write('{:<5} {}  {} I D\n'.format(staname, network, comp))
            stas_pick_dict[staname] = polstr

        else:
            pol = '-'
    fid.write('{}{:>16}\n'.format(str56, eventID))
    fid.close()

#==================================================================================================

def create_sta_reverse(folder_station, outfile):
    """
    staname is sorted
    start and end times are all zero
    """
    stanames = get_sta_names(folder_station)
    fid = open(outfile, 'w')
    for i in range(0, len(stanames)):
        fid.write('{:<5} 30000101 30000101\n'.format(stanames[i]))
    fid.close()

#==================================================================================================

def create_stations(folder_station, outfile):
    """
    sorted station file, with info of chan, lat, lon, elev, net
    example 2 use the old SCEDC format, there are only 4 letters for station names
    The current station names can have 5 letters, 
    """
    
    str1 = ''
    for i in range(0, 33):
        str1 = str1 + ' '
    str2 = ''
    for i in range(0, 23):
        str2 = str2 + ' '
        
    stanames = get_sta_names(folder_station)
    nsta = len(stanames)
    
    fid = open(outfile, 'w')
    for i in range(0, nsta):
        staname = stanames[i]
        filename_sta = glob.glob('{}/*{}*.txt'.format(folder_station, staname))        
        df_sta = pd.read_csv(filename_sta[0], sep='|', header=0,
                        usecols=[0, 1, 3, 4, 5, 6, 15, 16])
        chans = []                
        for j in range(0, df_sta.shape[0]):
            if df_sta['Channel'][j] in chans:
                continue                                        
            fid.write('{:>5} {}{}{:>9.5f} {:>10.5f} {:>5d}{}{}\n'.format(\
                    df_sta['Station'][j], df_sta['Channel'][j], str1, df_sta['Latitude'][j], df_sta['Longitude'][j], \
                    int(df_sta['Elevation'][j]), str2, df_sta['#Network'][j]))
            chans.append(df_sta['Channel'][j])
    fid.close()

#==================================================================================================

def create_stations_5char(folder_station, outfile):
    """
    sorted station file, with info of chan, lat, lon, elev, net
    example 2 use the old SCEDC format, there are only 4 letters for station names
    The current station names can have 5 letters, 
    """

    sta_file = f"{folder_station}/event_inventory.txt"
    
    df = pd.read_csv(sta_file, 
                    sep='|', 
                    header=0
                    ) 
    
    fid = open(outfile, 'w')
    str1 = ' ' * 37
    str2 = ' ' * 23

    for index, row in df.iterrows():

        starttime = row['StartTime']
        starttime = '{}/{}/{}'.format(starttime[0:4], starttime[5:7], starttime[8:10])

        endtime = row['EndTime']
        if type(endtime) != str:
            endtime = '3000/01/01'
        else:
            endtime = '{}/{}/{}'.format(endtime[0:4], endtime[5:7], endtime[8:10])

        fid.write('{}  {:<5} {}{}{:>9.5f} {:>10.5f} {:>5d} {} {}\n'.format(\
                row['#Network'], row['Station'], row['Channel'], str1, row['Latitude'], row['Longitude'], \
                int(row['Elevation']), starttime, endtime))
        chans = []
        chans.append(row['Channel'])
    fid.close()
    
#==================================================================================================

def get_sta_dict(folder_station, column_list):
    """
    #Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|
    SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime

    :param folder_station: contains all station files in txt format
    :param column_list: a list of the following column_list: 
                #Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|
                SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
    :return: a dictionary of station info
            key: station name
            value: a list of station info
    """
    sta_file = f"{folder_station}/event_inventory.txt"
    sta_loc = {}
    
    df = pd.read_csv(sta_file, 
                    sep='|', 
                    header=0
                    )   
    df.drop_duplicates(subset=['Station'], inplace=True)
    
    for index, row in df.iterrows():
        
        sta_loc[row['Station']] = [row['Latitude'], row['Longitude']]

    return sta_loc

#==================================================================================================

def get_sta_names(folder_station):
    sta_file = f"{folder_station}/event_inventory.txt"
    sta_loc = {}
    
    df = pd.read_csv(sta_file, 
                    sep='|', 
                    header=0
                    )   
    df.drop_duplicates(subset=['Station'], inplace=True)
    stanames = df['Station'].values

    return np.sort(stanames)

#==================================================================================================

def make_input_file(folder_station, event_id, outfile):
    """
    Makes an input file for HASH following hash input type 2.
    input:
        folder_station: folder where the event data is stored
        event_id: event id
        outfile: name of the output file
    """
    with open(outfile, 'w') as fid:
        fid.write(f"""{folder_station}/{event_id}.stations
{folder_station}/{event_id}.reverse
{folder_station}/{event_id}.phase
{folder_station}/{event_id}.out
{folder_station}/{event_id}.out2
8
90
60
5
30
500
0.1
120
45
0.25
5
./HASH/vz.MTJ
./HASH/vz.MTJ1
./HASH/vz.MTJ2
./HASH/vz.MTJ3
./HASH/vz.MTJ4"""
        )

#==================================================================================================