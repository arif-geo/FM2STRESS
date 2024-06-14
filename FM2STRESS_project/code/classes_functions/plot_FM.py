import pandas as pd
import numpy as np
import pygmt


# To plot focal mechanism on map we need the following:
# 1. Focal mechanism parameters: strike, dip, rake
# 2. Event location: elat, elon, edep and magnitude
# 3. station locations and polarities

# read the SKHASH output file to get the focal mechanism parameters
def get_FM_params(skhash_out_file, event_id=None):

    """
    input:
        skhash_out_file: SKHASH output file [path]
        event_id: event id [single value]
    output:
        if event_id is provided:
            strike, dip, rake [float]: focal mechanism parameters for the event
            fm_df [df]: all focal mechanism details
        else:
            fm_df [df]: all focal mechanism details
    """
    fm_df = pd.read_csv(skhash_out_file) # standard csv file so no need to specify any delimiter

    if event_id is not None:
        # get focal mechanism details for the event
        strike, dip, rake, fqual = fm_df[fm_df['event_id'] == event_id][['strike', 'dip', 'rake', 'quality']].values[0]
        return strike, dip, rake, fqual, fm_df
    
    # if mutiple solution is found, filter by 'quality' column [A: best, D: worst]
    fm_df = fm_df.sort_values(by=['event_id', 'quality'], ascending=[True, True])
    fm_df = fm_df.drop_duplicates(subset='event_id', keep='first')
    fm_df = fm_df.sort_values(by='quality').reset_index(drop=True)
    return fm_df # return all focal mechanism details


def get_sta_lat_lon_pol(event_id, skhash_pol_file, sta_inv_file):

    """
    add sta lat, lon to the skhash polarity file 
    input: 
        event_id: event id  [single value]
        skhash_pol_file: SKHASH polarity file [path]
        sta_inv_file: station inventory file txt [path]
    output:
        event_sta_pol_df [df]: same as SKHASH polarity file but with additional columns for station latitude and longitude
        elat, elon, edep [float]: event latitude, longitude and depth [single value]
    """
    # read skhash polarity file
    sta_pol_df = pd.read_csv(skhash_pol_file)

    # sta inventory file
    sta_inv_df = pd.read_csv(sta_inv_file,
                            header=0, sep='|', usecols=['Station', 'Latitude', 'Longitude'],
                            ).drop_duplicates(subset='Station').reset_index(drop=True)
    
    event_sta_pol_df = sta_pol_df[sta_pol_df['event_id'] == event_id]

    # add new columns for station latitude and longitude
    event_sta_pol_df.insert(5, 'sta_lat', np.nan)
    event_sta_pol_df.insert(6, 'sta_lon', np.nan)
    
    # for each station in the event, get the latitude and longitude from the station inventory file
    for i, row in event_sta_pol_df.iterrows():
        sta = row['station']
        try:
            event_sta_pol_df.loc[i, 'sta_lat'] = sta_inv_df.loc[sta_inv_df['Station'] == sta, 'Latitude'].values[0]
            event_sta_pol_df.loc[i, 'sta_lon'] = sta_inv_df.loc[sta_inv_df['Station'] == sta, 'Longitude'].values[0]
        except:
            pass

    elat, elon, edep = event_sta_pol_df[['origin_latitude','origin_longitude','origin_depth_km']].values[0]
    
    return event_sta_pol_df, elat, elon, edep 


def plot_FM_stations(
    event_id,
    event_sta_pol_df, # with station lat and lon
    elat, elon, edep, # event location
    strike, dip, rake, # focal mechanism parameters
    magnitude: float = None,
    region: list = [-180, 180, -90, 90],
    projection: str = 'M12c', # M12c: mercator cylindrical
    ):

    """
    """

    fig = pygmt.Figure()

    # generate a basemap near Washington state showing coastlines, land, and water
    # pygmt.config(MAP_ANNOT_OBLIQUE="lat_parallel", MAP_FRAME_TYPE="plain")
    with pygmt.config(MAP_FRAME_TYPE="plain"):
        # add basemap
        fig.basemap(
            region=region,
            projection="M12c",
            frame=["WSne", "xa.5f+lLongitude", "ya.5f+lLatitude"]
        )

    # add coastlines
    fig.coast(
        region=region,
        projection="M12c",
        land="grey",
        water="lightblue",
        shorelines=True,
        resolution="f",
    )


    # store focal mechanisms parameters in a dict
    focal_mechanism = dict(strike=strike, dip=dip, rake=rake, magnitude=5.0)

    # pass the focal mechanism data to meca in addition to the scale and event location
    fig.meca(focal_mechanism, scale="1c", longitude=elon, latitude=elat, depth=edep)

    # plot the stations with polarities [ color based on polarity, black: positive, white: negative]
    pygmt.makecpt(cmap="grayC", series=[-1, 1], reverse=False) # default: -1: black  1: white

    fig.plot(
        x=event_sta_pol_df.sta_lon,
        y=event_sta_pol_df.sta_lat,
        style="t0.3c", 
        fill= event_sta_pol_df.p_polarity,
        cmap=True,
        pen='black',
    )
    fig.colorbar(frame=["x+lPolarity", "y+lm"])


    # for i, row in event_sta_pol_df.iterrows():
    #     if row.p_polarity == 1: # positive polarity, compressional wave, filled circle
    #         fig.plot(x=row.sta_lon, y=row.sta_lat, style="t0.3c", fill='black', pen='black')
    #     elif row.p_polarity == -1: # negative polarity, dilatational wave, empty circle
    #         fig.plot(x=row.sta_lon, y=row.sta_lat, style="t0.3c", fill='white', pen='black')
    #     else: continue

    # plot the station names
    fig.text(x=event_sta_pol_df.sta_lon, y=event_sta_pol_df.sta_lat-0.06,text=event_sta_pol_df.station, font='8p')
    # fig.text(x=event_sta_pol_df.sta_lon+0.08, y=event_sta_pol_df.sta_lat,text=event_sta_pol_df.p_polarity, font='8p')

    # remove corner xlabels and ylabels
    

    return fig   


