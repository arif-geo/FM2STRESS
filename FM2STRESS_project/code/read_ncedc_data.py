import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pygmt
from scipy.interpolate import griddata
import pdb
from obspy.clients import fdsn
from obspy import UTCDateTime,read,Trace,Stream


def read_eq_catalog(filename):
    fmt = '%Y/%m/%d %H:%M:%S.%f'
    df = pd.read_csv(filename, sep='\s+', skiprows=15, header=None, \
            names=['Date', 'Time', 'Lat', 'Lon', 'Depth', 'Mag', 'Magt', 'Nst', 'Gap', 'Clo', 'RMS', 'SRC', 'EventID'], \
            dtype={'Date':str, 'Time':str, 'Lat':float, 'Lon':float, 'Depth':float, 'Mag':float, \
            'Magt':str, 'Nst':int, 'Gap':int, 'Clo':int, 'RMS':float, 'SRC':str, 'EventID':int})
    eventID, elat, elon, edep, mag = df['EventID'].values, df['Lat'].values, df['Lon'].values, df['Depth'].values, df['Mag'].values
    Nst, Gap, RMS = df['Nst'].values, df['Gap'].values, df['RMS'].values
    etime = []
    for i in range(0, df.shape[0]):
        datestr = '{} {}0000'.format(df['Date'].values[i], df['Time'].values[i])
        etime.append(datetime.datetime.strptime(datestr, fmt))
    return eventID, elat, elon, edep, mag, etime, Nst, Gap, RMS


def read_NCEDC_Focal_Mechnism(filename):
    fid = open(filename, 'r')
    lines = fid.readlines()
    fid.close()
    N = len(lines)
    print(N)
    eventID, NPS = np.zeros((N, )).astype(int), np.zeros((N, )).astype(int)
    elat, elon, edep, mag = np.zeros((N, )), np.zeros((N, )), np.zeros((N, )), np.zeros((N, ))
    Gap, RMS, Herr, Verr = np.zeros((N, )), np.zeros((N, )), np.zeros((N, )), np.zeros((N, ))
    strike, dip, rake, misfit, Npolarity = np.zeros((N, )), np.zeros((N, )), np.zeros((N, )), np.zeros((N, )), np.zeros((N, )).astype(int)
    etime = []
    fmt = '%Y/%m/%d %H:%M:%S.%f'
    for i in range(0, N):
        line = lines[i]
        datestr = '{}/{}/{} {}:{}:{:02d}.{}0000'.format(\
            line[0:4], line[4:6], line[6:8], \
            line[9:11], line[11:13], int(line[14:16]), line[17:19])
        etime.append(datetime.datetime.strptime(datestr, fmt))
        elat[i] = float(line[19:22]) + float(line[23:28])/60
        elon[i] = float(line[29:32]) + float(line[33:38])/60
        edep[i] = float(line[38:45])
        mag[i] = float(line[47:52])
        NPS[i] = int(line[52:55])
        Gap[i] = float(line[55:59])
        RMS[i] = float(line[64:69])
        Herr[i] = float(line[69:74])
        Verr[i] = float(line[74:79])
        strike[i] = float(line[83:86])
        dip[i] = float(line[87:89])
        rake[i] = float(line[89:93])
        misfit[i] = float(line[95:99])
        Npolarity[i] = int(line[100:103])
        eventID[i] = int(line[133:141])
    return eventID, elat, -elon, edep, mag, etime, strike, dip, rake, misfit, Npolarity, NPS, Gap, RMS, Herr, Verr 
    
    
def combine_picks():
    fid = open('picks_2008_2024.txt', 'w')
    for i in range(2008, 2024):
        fid1 = open('Input/{}_picks.txt'.format(i), 'r')
        lines = fid1.readlines()
        fid1.close()
        for j in range(0, len(lines)):
            fid.write(lines[j])
    fid.close()

def Hypoinverse_to_SKHASH():
    """
    event_id, station, network, location, channel, p_polarity, takeoff, takeoff_uncertainty, azimuth, azimuth_uncertainty
    123456,   ST000,   XX,      --,       EHZ,     -1.0,       53.5,   0.1,                 75.0,   0.1
    123456,   ST001,   XX,      --,       EHZ,     -1.0,       53.5,   0.1,                 69.7,   0.1
    123456,   ST002,   XX,      --,       EHZ,     -1.0,       53.5,   0.1,                 64.4,   0.1
    123456,   ST003,   XX,      --,       EHZ,     -1.0,       53.5,   0.1,                 59.1,   0.1
    123456,   ST004,   XX,      --,       EHZ,     -1.0,       53.5,   0.1,                 53.9,   0.1

    """
    filename = '/Users/gongjian/Documents/Research/Grad/Arif/QuakeML/HYPOINVERSE_ARCHIVE/picks_2008_2024.txt'
    fid = open(filename, 'r')
    lines = fid.readlines()
    fid.close()
    event_index = []
    for i in range(0, len(lines)):
        if len(lines[i])>130:
            event_index.append(i)
    event_index.append(len(lines))

    # output
    fid = open('polarity.csv', 'w')
    fid.write('event_id,station,network,location,channel,p_polarity\n')
    for i in range(0, len(event_index) - 1):
    # for i in range(0, 10):
        j1, j2 = event_index[i], event_index[i+1]
        # read event header
        line = lines[j1]
        eyear, emonth, eday, ehour, eminute = int(line[0:4]), int(line[4:6]), int(line[6:8]), int(line[8:10]), int(line[10:12])
        esec = float(line[12:16])/100
        elat_int = int(line[16:18])
        elat_minute = float(line[19:23])/100
        elon_int = int(line[23:26])
        elon_minute = float(line[27:31])/100
        edep = float(line[31:36])/100
        e_hori_err = float(line[85:89])/100
        e_vert_err = float(line[89:93])/100
        emag = float(line[147:150])/100
        eventID = int(line[136:146])   

        for j in range(j1+1, j2-1):
            line = lines[j]
            staname = line[0:5].strip()
            network = line[5:7]
            comp = line[8:12]
            if line[14] == 'P':
                P_remark = line[13]
                if line[15] == 'U':
                    fid.write('{},{},{},--,{},1.0\n'.format(eventID, staname.strip(), network, comp.strip()))
                elif line[15] == 'D':
                    fid.write('{},{},{},--,{},-1.0\n'.format(eventID, staname.strip(), network, comp.strip()))
    fid.close()
                        


def create_eq_catalog():
    """
    Create eq catalog for SKHASH format
    
    time, latitude, longitude, depth, horz_uncert_km, vert_uncert_km, mag, event_id
    2000-01-01 00:00:00.000, 37.412116, -122.059357, 12.345, 0.2, 0.4, 2.3, 123456
    2001-01-01 00:00:00.000, 37.412116, -122.059357, 12.345, 0.2, 0.4, 2.3, 654321
    """
    
    eventID, elat, elon, edep, mag, etime, strike, dip, rake, misfit, Npolarity, NPS, Gap, RMS, Herr, Verr = \
    read_FM('Input/focal_mechanism_2008_2024.txt')
    fmt = '%Y-%m-%d %H:%M:%S.%f'
    fid = open('MTJ_catalog.csv', 'w')
    fid.write('time, latitude, longitude, depth, horz_uncert_km, vert_uncert_km, mag, event_id\n')
    for i in range(0, len(eventID)):
        etime_str = datetime.datetime.strftime(etime[i], fmt)[0:-3]     
        if edep[i] < 0:
            fid.write('{}, {:.6f}, {:.6f}, {:.3f}, {:.2f}, {:.2f}, {:.1f}, {:d}\n'.format(\
                etime_str, elat[i], elon[i], 0.1, Herr[i], Verr[i], mag[i], eventID[i]))
        else:
            fid.write('{}, {:.6f}, {:.6f}, {:.3f}, {:.2f}, {:.2f}, {:.1f}, {:d}\n'.format(\
                etime_str, elat[i], elon[i], edep[i], Herr[i], Verr[i], mag[i], eventID[i]))
    fid.close()

def create_stations():
    """
    Create stations for SKHASH format
    Directly query from Client

    station,location,channel,latitude,longitude,elevation
    AL1,--,DPZ,38.83822,-122.88345,704
    AL1,--,DPE,38.83822,-122.88345,704
    """

    outfile = 'MTJ_stations.csv'
    fid = open(outfile, 'w')
    fid.write('station,location,channel,latitude,longitude,elevation\n')

    STATION1 = 'https://service.iris.edu/fdsnws/station/1'
    st_client1 = fdsn.client.Client('http://service.iris.edu',
                                   service_mappings={
                                   'station': STATION1
                                   },
                                   debug=False
                                   )
    STATION2 = 'https://service.ncedc.org/fdsnws/station/1'
    st_client2 = fdsn.client.Client('http://service.ncedc.org',
                                   service_mappings={
                                   'station': STATION2
                                   },
                                   debug=False
                                   )
    sta_df1 = pd.read_csv('polarity.csv')
    sta_net = []
    for i in range(0, sta_df1.shape[0]):
        sta_net.append(sta_df1['station'][i]+'-'+sta_df1['network'][i])
    unique_sta_net = np.unique(sta_net)
    starttime = UTCDateTime(2008,1,1)
    endtime = UTCDateTime(2024,1,1)
    for i in range(0, len(unique_sta_net)):
        items = unique_sta_net[i].split('-')
        try:
            inventory = st_client1.get_stations(network=items[1], station=items[0],\
               starttime=starttime,endtime=endtime)
        except:
            print('{} No IRIS'.format(unique_sta_net[i])) 
            inventory = st_client2.get_stations(network=items[1], station=items[0],\
               starttime=starttime,endtime=endtime)

        if len(inventory.networks[0].stations) > 1:
            print('{} >1 stations, checking...'.format(unique_sta_net[i])) 
            slat, slon, selev = [], [], []
            for j in range(0, len(inventory.networks[0].stations)):
                slat.append(inventory.networks[0].stations[j].latitude)
                slon.append(inventory.networks[0].stations[j].longitude)
                selev.append(inventory.networks[0].stations[j].elevation)
            if (np.max(slat)-np.min(slat)>0.000001) or (np.max(slon)-np.min(slon)>0.000001):
                print('{} moved'.format(unique_sta_net[i])) 
                print(slat)
                print(slon)
                print(selev)
                slat, slon, selev = slat[0], slon[0], selev[0] 
            else:
                slat, slon, selev = slat[0], slon[0], selev[0]        
        else:
            slat, slon, selev = inventory.networks[0].stations[0].latitude, \
                            inventory.networks[0].stations[0].longitude, \
                            inventory.networks[0].stations[0].elevation
        temp_df = sta_df1[(sta_df1['station']==items[0]) & (sta_df1['network']==items[1])]
        a=temp_df['channel']
        chans = np.unique(a)
        for j in range(0, len(chans)):
            fid.write('{},--,{},{:.5f},{:.5f},{:d}\n'.format(\
                    items[0], chans[j], slat, slon, int(selev)))
    fid.close()




# 1
# combine the yearly picks into one single file
combine_picks()
# 2
# convert the picks file in Hypoinverse format to SKHASH format
Hypoinverse_to_SKHASH()
# 3
# list all the stations appearred in the Hypoinverse 
create_stations()

      
