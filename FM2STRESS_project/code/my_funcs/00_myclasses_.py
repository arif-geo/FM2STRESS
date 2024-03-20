# create a class to convert file formats 
class FileFormatConversions:
    def __init__(self, file_path):
        self.file_path = file_path
    
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

        import pandas as pd
        import numpy as np

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
                sta, loc, cha = line[4].split('.')[1], line[4].split('.')[2], line[4].split('.')[3]
                phase = line[8]
                pol = line[9]
                
                # add to pol_df
                pol_df.loc[index] = [event_id, i+1, sta, loc, cha, pol, elat, elon, edep]
                index += 1

        
        # return the dataframe
        return pol_df[pol_df.p_polarity != 'None']
        # return pol_df
        # pol_df[pol_df.p_polarity != 'None'].to_csv(f'{output_path}/00_pol_consensus_master_polarity.csv', index=False)
        