import os
import numpy as np
import pandas as pd

class NCEDC_HypoInverse_reader:
    def __init__(self, ncedc_hypoinv_file=None):
        self.ncedc_hypoinv_file = ncedc_hypoinv_file

    ############################
    ############################

    def NCEDC_picks_2_dict(self, ncedc_pick_dir, file):
        """
        Function to convert NCEDC picks file into a dictionary
        dict keys: event_id
        dict values: list of lists with the following columns:
            ['event_id', 'etime', 'elat', 'elon', 'edep', 'emag', 'station_id', 'phase_type', 'phase_polarity', 'phase_time'])

        """
        my_dict = {}
        current_event = None
        current_event_params = []
        line_len = []

        with open (f'{ncedc_pick_dir}/{file}', 'r') as f:
            lines = f.readlines()
            
            for i, line in enumerate(lines): 
                line_len.append(len(line))
                ### Separate event header and station picks by line length ###
                if len(line) >= 170: # event header
                    if current_event is not None:   
                        my_dict[current_event] = current_event_params

                    eymd = line[0:4] + '-' + line[4:6] + '-' + line[6:8]
                    edts = line[8:10] + ':' + line[10:12] + ':' + line[12:14] + '.' + line[14:16]
                    etime = eymd + 'T' + edts
                    elat = float(line[16:18].strip()) + float(line[19:23].strip())/100/60 # /100: F4.2, /60: min to deg
                    elon = float(line[23:27].strip()) + float(line[27:31].strip())/100/60
                    edep = np.float32(line[31:36].strip())/100
                    eid = 'nc' + line[136:146].strip()
                    emag = float(line[147:150].strip())/100 # F3.2

                    evparams = [eid, etime, elat, elon, edep, emag]
                    current_event = eid
                    current_event_params = [] # reset for the next event

                elif len(line) == 121: # station picks
                    stcode = line[0:5].strip()
                    net = line[5:7].strip()
                    cha = line[9:12].strip()
                    phase_type = line[14:15].strip()
                    polarity = line[15:16].strip()
                    sec_str = line[29:34].strip()
                    if sec_str == '' or phase_type == '':
                        continue
                    sec = float(sec_str)/100
                    phase_time = f'{line[17:21]}-{line[21:23]}-{line[23:25]}T{line[25:27]}:{line[27:29]}:{sec}'

                    evstnparams = evparams + [f'{net}.{stcode}..{cha}', phase_type, polarity, phase_time]
                    current_event_params.append(evstnparams)

                    if i == len(lines)-2: # last line -1 , so save the last event
                        my_dict[current_event] = current_event_params
                
                else:
                    # print(line, len(line))
                    pass
        
        return my_dict

    ############################
    ############################

    def picks_dict_to_df(self, picks_dict):
        """
        Function to convert dictionary from "NCEDC_picks_2_dict" func to a pandas dataframe
        """
        df_list = []
        # put the dict into a dataframe
        for k, v in picks_dict.items():
            temp_df = pd.DataFrame(
                v, 
                columns=['event_id', 'etime', 'elat', 'elon', 'edep', 'emag', 'station_id', 'phase_type', 'phase_polarity', 'phase_time'])
            df_list.append(temp_df)

        # combine the dataframes into one
        master_df = pd.concat(df_list)
        # master_df['phase_polarity'] = master_df['phase_polarity'].apply(
        #     lambda x: 1 if x == 'U' else -1 if x == 'D' else np.nan)

        # convert lat and lon to float and round to 7 decimal places and times to datetime
        master_df['elat'] = master_df['elat'].astype(float)
        master_df['elon'] = master_df['elon'].astype(float) * -1
        master_df['etime'] = pd.to_datetime(master_df['etime'])
        master_df['phase_time'] = pd.to_datetime(master_df['phase_time'])

        # filter out events outside of the region of interest
        master_df = master_df[
            (master_df['elat'] >= 39.75) & (master_df['elat'] <= 41.5) 
            & (master_df['elon'] >= -125.5) & (master_df['elon'] <= -123)]

        # keep only picks ending with 'Z' or '3'
        master_df = master_df[master_df['station_id'].str.endswith(('Z', '3'))]
        # Polarity U/D to 1/-1
        master_df['phase_polarity'] = master_df['phase_polarity'].apply(
            lambda x: 1 if x == 'U' else -1 if x == 'D' else np.nan)

        return master_df

    ############################
    ############################

    def df_to_pyrocko_marker(self, picks_df, output_dir, file_name, color_row=True):
        """
        Function to convert a pandas dataframe to a pyrocko marker
        df with columns: ['event_id', 'etime', 'elat', 'elon', 'edep', 'emag', 'station_id', 'phase_type', 'phase_polarity', 'phase_time']
        ** new: color_row: 'color' column in the dataframe
            to identify the different sources of the picks
        """
        import random, string
        random.seed(42)

        pyrocko_fmt = []
        # convert the dataframe to PyRocko format text file
        for ig, groupdf in picks_df.groupby('event_id'):

            # generate 27 char random string
            random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=27)) + '='

            # event line
            etime_formatted = pd.to_datetime(groupdf.iloc[0]['etime']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-2]
            event_line = f"event: {etime_formatted}  0 {random_string}   "
            event_line += f"{groupdf.iloc[0]['elat']:.7f} {groupdf.iloc[0]['elon']:.7f}         "
            event_line += f"{groupdf.iloc[0]['edep']:.2f} {groupdf.iloc[0]['emag']:.2f} None  {groupdf.iloc[0]['event_id']} None\n" # dep, mag, ???, event_id, ???
            
            pyrocko_fmt.append(event_line)
            # print((event_line))

            for i, row in groupdf.iterrows(): 
                phase_time_obj = pd.to_datetime(row['phase_time'])
                color = int(row.color) if color_row else '0'
                part1 = f"phase: {phase_time_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-2]}  {color} "
                part2 = f"{row.station_id.ljust(15)}  {random_string} {etime_formatted} "
                polarity = int(row.phase_polarity) if row.phase_polarity in [-1, 1] else 'None'
                part3 = f"{row.phase_type}          {str(polarity).rjust(4)} False\n"

                phase_line = part1 + part2 + part3
                
                pyrocko_fmt.append(phase_line)

        pyrocko_fmt = '# Snuffler Markers File Version 0.2\n' + ''.join(pyrocko_fmt)
        with open(f'{output_dir}/{file_name}', 'w') as f:
            # f.write('# Snuffler Markers File Version 0.2\n')
            f.writelines(pyrocko_fmt)
        return pyrocko_fmt

    ############################

    # FPFIT FORMAT to CSV
    
    ############################
def NCEDC_FPFIT_FM_2_csv(ncedc_fm_file, header_lines=0):
    """
    INPUT:
        For documentation on the NCEDC FPFIT format, see:
        https://ncedc.org/pub/doc/cat5/ncsn.mech.txt

        To download the NCEDC FPFIT solutions, see:
        https://ncedc.org/ncedc/catalog-search.html (select "Mechanism Catalog" 
            and under "Output Format" select "FPFIT")
    OUTPUT:
        csv file with columns similar to SKHASH output.
        First 4 columns are exactly same as SKHASH output.
    """

    with open (ncedc_fm_file, 'r') as f:
        lines = f.readlines()
        lines = lines[header_lines:]       # skip the header

        # file_data={}
        file_data = []
        for i, line in enumerate(lines):
            if (len(line)) < 141:
                continue

            # Determine the sign of the latitude and longitude
            n = -1 if line[19] == 'S' else 1
            e = 1 if line[32] == 'E' else -1
            
            # Extract the data into a dictionary
            line_data = {
                'event_id': line[131:141].strip(),
                'strike': float(line[83:86].strip()) + 90 if float(line[83:86].strip()) < 180 else float(line[83:86].strip()) - 90,
                'dip': float(line[87:89].strip()),
                'rake': line[89:93].strip(),
                'time': f'{line[0:4].strip()}:{line[4:6].strip()}:{line[6:8].strip()}T{line[9:11].strip()}:{line[11:13].strip()}:{line[14:19].strip()}',
                'latitude': (float(line[19:22].strip()) + float(line[23:25].strip())/60) * n,
                'longitude': (float(line[28:32].strip()) + float(line[33:35].strip())/60) * e,
                'depth': line[38:45].strip(),
                'mag': float(line[47:52].strip()),
                'max_agap': line[55:59].strip(),
                'dip_azimuth': line[83:86].strip(),
                'misfit': line[95:99].strip(),
                'n_fmo': line[100:103].strip(), # Number of first motion observations used in solution
                'misfit_w_90ci': line[104:109].strip(),
                'max_strike_90ci': line[121:123].strip(),
                'max_dip_90ci': line[124:126].strip(),
                'max_rake_90ci': line[127:129].strip(),
                'mult_solution_flag': 1 if line[130].strip() == '*' else 0,
            }
            # file_data[i] = line_data
            file_data.append(line_data)

    # ncedc_fm_df = pd.DataFrame.from_dict(file_data, orient='index')
    ncedc_fm_df = pd.DataFrame(file_data)
    ncedc_fm_df['time'] = pd.to_datetime(ncedc_fm_df['time'], format='%Y:%m:%dT%H:%M:%S.%f')
    ncedc_fm_df['latitude'], ncedc_fm_df['longitude'] = ncedc_fm_df['latitude'].round(5), ncedc_fm_df['longitude'].round(5)

    return ncedc_fm_df



