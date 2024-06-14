

import pandas as pd
from classes_functions.NCEDC_files_converter import NCEDC_FPFIT_FM_2_csv

ncedc_fm_file = '../data/NCEDC_FM_solutions/2008-24_39.75-41.5_125.5-123_fpfit.txt'
ncedc_fm_df = NCEDC_FPFIT_FM_2_csv(ncedc_fm_file)
ncedc_fm_df.to_csv('../data/NCEDC_FM_solutions/2008-24_39.75-41.5_125.5-123_fpfit.csv', index=False)
print(f'File saved to ../data/NCEDC_FM_solutions/2008-24_39.75-41.5_125.5-123_fpfit.csv')
