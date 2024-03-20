# This file contains functions to convert marker file to SKHASH format
import os
import glob
import pandas as pd
import numpy as np


############ function to convert marker file to SKHASH format ############
def pyrocko_marker2skhash_pol(markerfile_path, output_path='.'):
    """
    This function creates a polarity file for SKHASH format input data
    input: 
        markerfile_path: path to the markerfile [master marker file]
        output_path: path to write the output file
    output:
        pol_consensus.csv file written to ouput folder
            columns: [event_id,event_id2,station,location,channel,p_polarity,origin_latitude,origin_longitude,origin_depth_km]
    """

    # create an empty dataframe with column names:
    pol_df = pd.DataFrame(
        columns=['event_id','event_id2','station','location','channel','p_polarity','origin_latitude','origin_longitude','origin_depth_km']
    )

    # empty dictionary to store the markerfile data
    mydict = {}

    # read the markerfile line by line
    with open(markerfile_path, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            if line.split()[0] == "event:":
                event_id = line.split()[-2]
                key = line
                mydict[key] = []
            else:
                mydict[key].append(line.split())
        
    # loop over each event_id
    index = 0
    for i, event_line in enumerate(mydict.keys()):
        event_id = event_line.split()[-2]
        elat, elon, edep = event_line.split()[5], event_line.split()[6], event_line.split()[7]
        emag = event_line.split()[8]
        for j, line in enumerate(mydict[event_line]):
            if not line[0] == "phase:":
                continue
            
            # get the station, location, channel, phase, polarity
            sta, cha = line[4].split('.')[1], line[4].split('.')[3]
            loc = line[4].split('.')[2]
            # loc = '--' if loc == '' else loc
            phase = line[8]
            pol = line[9]
            
            # add to pol_df
            pol_df.loc[index] = [event_id, i+1, sta, loc, cha, pol, elat, elon, edep]
            index += 1


    # replace the empty location with '--'
    # convert all empty cells to np.nan
    pol_df = pol_df.replace('', np.nan)
    pol_df.location = pol_df.location.fillna('--')

    # return the dataframe
    return pol_df[pol_df.p_polarity != 'None']
    # return pol_df
    # pol_df[pol_df.p_polarity != 'None'].to_csv(f'{output_path}/00_pol_consensus_master_polarity.csv', index=False)

############ END OF FUNCTION ####################
#
############ POLARITY ###########################
#
#################################################

"""
**pyrocko marker to skshash pol_concensus for individual event marker file.
This is my old  filing system. I make 1 marker file for one event. Now, i have a master marker file for all events.**
"""
def make_skhash_polarity_file(markerfile_path, catalog_path, output_path):
    ##### old #####
    """
    This function creates a polarity file for SKHASH format input data
    input: 
        markerfile_path: path to the markerfile [master marker file]
        catalog_path: path to the catalog file [master catalog file]
    output:
        pol_consensus.csv file written to ouput folder
    """

    # create an empty dataframe with column names:
    # [event_id,event_id2,station,location,channel,p_polarity,origin_latitude,origin_longitude,origin_depth_km]
    pol_df = pd.DataFrame()

    # loop over each event_id
    for i, event_id in enumerate(event_ids):

        # read the picks.txt file
        # col4 has NN.SSSSS.LL.CCC format (network.station.location.channel)
        picks_df = pd.read_csv(
            f"{data_path}/{event_id}/event_{event_id}_picks.txt",
            skiprows=1,         # skip the first row
            sep='\s+',          # delimiter
            header=None,        # no header
            usecols=[4,8,9],    # only read columns 4,8,9,
            names=['stns', 'phase', 'polarity']
            )
        # get 'origin_latitude','origin_longitude','origin_depth_km' from original catalog file
        ## or we can also use best_location file we created using misfit calculation
        catalog_df = pd.read_csv(catalog_path, sep=',', header=0, 
                                usecols=['latitude','longitude','depth', 'id'],
                                )
        catalog_df = catalog_df[catalog_df['id'] == event_id]

        # create a temporary dataframe with columns [event_id,event_id2,station,location,channel,p_polarity]
        temp_df = pd.DataFrame(columns=['event_id','event_id2','station','location','channel','p_polarity'])
        
        temp_df['station'] = picks_df['stns'].str.split('.').str[1]
        temp_df['location'] = picks_df['stns'].str.split('.').str[2]
        temp_df['location'].replace('', '--', inplace=True)

        temp_df['channel'] = picks_df['stns'].str.split('.').str[3]
        temp_df['p_polarity'] = picks_df['polarity']
        temp_df['event_id'] = event_id
        temp_df['event_id2'] = i+1
        temp_df['origin_latitude'] = format(catalog_df['latitude'].values[0], '.5f')
        temp_df['origin_longitude'] = format(catalog_df['longitude'].values[0], '.5f')
        temp_df['origin_depth_km'] = catalog_df['depth'].values[0]

        # append the temp_df to pol_df
        pol_df = pd.concat([pol_df, temp_df], ignore_index=True)

        # drop rows with nan values in p_polarity column
        pol_df.dropna(subset=['p_polarity'], inplace=True)
        

    # write the pol_df dataframe to a csv file
    print(f"{'='*10} Writing the `pol_consensus.csv` file to {output_path}")
    pol_df.to_csv(f'{output_path}/pol_consensus.csv', index=False)

############ END OF FUNCTION ####################
#
############ MAKE SKHAH STATION FILE ############
#
#################################################

def make_skhash_station_file(
    output_path=None,
    client_list = ['IRIS', 'NCEDC'], 
    starttime = "2008-01-01", 
    endtime = "2023-01-01",
    region = [-128, -122.5, 39, 42],
    channel = 'HH*,BH*,HN*,EH*',
    given_inventory = None
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
        import obspy
        from obspy import read, UTCDateTime, read_inventory, Inventory
        from obspy.clients.fdsn import Client 

        # get station inventory
        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime)

        minlongitude = region[0]
        maxlongitude = region[1]
        minlatitude = region[2]
        maxlatitude = region[3]

        merged_inventory = Inventory()
        for iclient in client_list:
            print(iclient)
            client = Client(iclient)
            try:
                try:
                    inv = client.get_stations(
                        starttime = starttime,
                        endtime = endtime,
                        minlatitude = minlatitude,
                        maxlatitude = maxlatitude,
                        minlongitude = minlongitude,
                        maxlongitude = maxlongitude,
                        channel = channel,
                        level = 'response'
                    )
                    merged_inventory.networks.extend(inv.networks)
                except:
                    pass
            except:
                print(f"Failed to get inventory from {iclient}")
                continue

        # temorarily write merged inventory to a file
        merged_inventory.write("mtj_merged_stations_temp.txt", format="STATIONTXT")
        given_inventory = "mtj_merged_stations_temp.txt"

    else:
        print(f"Using the provided merged inventory file: {given_inventory}")
        pass

    # read the merged inventory file
    inv_df = pd.read_csv(given_inventory, sep='|', skiprows=1,
                        usecols=[1, 2, 3, 4, 5, 6],
                        names=['station','location','channel','latitude','longitude','elevation'],
                        dtype={'station': str,'location': str, 'channel': str, 'latitude': float, 'longitude': float,
                                'elevation': float},    
                        )
    # replace the empty location with '--'
    # inv_df['location'] = inv_df['location'].fillna('--')
    inv_df.location = inv_df.location.fillna('--')

    # sort by station and write the station file
    # print(f"{'='*10} Writing the `station.csv` file to {output_path}")
    inv_df.sort_values(by=['station']).reset_index(drop=True)
    inv_df.drop_duplicates(subset=['station', 'location', 'channel'], keep='first', inplace=True) 

    # write the inv_df dataframe to a csv file
    # inv_df.to_csv(f'{output_path}/station.csv', index=False)
    return inv_df

    # delete the temporary file
    if os.path.exists("mtj_merged_stations_temp.txt"):
        os.remove("mtj_merged_stations_temp.txt")


############ END OF FUNCTION ####################
############ reverse file #######################
#################################################

def make_skhash_reverse_file(polarity_file, output_path=None):
    """
    make sure the polarity_file has been created in skhash format
    """

    df = pd.read_csv(polarity_file, header=0)
    
    reverse_df = pd.DataFrame()
    reverse_df['station'], reverse_df['location'], reverse_df['channel'] = df['station'], df['location'], df['channel']
    reverse_df.drop_duplicates(subset=['station'], inplace=True)

    # put dummy start and end time
    reverse_df['start_time'] = "1900-01-01"
    reverse_df['end_time'] = "2200-01-01"

    # print(f"{'='*10} Writing the `reverse.txt` file to {output_path}")
    # reverse_df.to_csv(f'{output_path}/reverse.csv', index=False)
    return reverse_df