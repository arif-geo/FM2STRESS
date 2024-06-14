#*************************************************************************#
#                                                                         #
#  script INPUT_PARAMETERS                                                #
#                                                                         #
#  list of input parameters needed for the inversion                      #
#                                                                         #
#*************************************************************************#
import numpy as np
import os

### NOTE: do not remove r before strings (r'filename'), to safely use
#         backslashes in filenames

#--------------------------------------------------------------------------
# input file with focal mechnaisms
#--------------------------------------------------------------------------
input_file = r'../Data/MTJ_manual_mechanisms.dat'

#--------------------------------------------------------------------------
# output file with results
#--------------------------------------------------------------------------
output_file = r'../Output/MTJ_manual_stress_output'

# ASCII file with calculated principal mechanisms
principal_mechanisms_file = r'../Output/MTJ_manual_principal_mechanisms'

#-------------------------------------------------------------------------
# accuracy of focal mechansisms
#--------------------------------------------------------------------------
# number of random noise realizations for estimating the accuracy of the
# solution
N_noise_realizations = 100
# =============================================================================
# 
# estimate of noise in the focal mechanisms (in degrees)
# the standard deviation of the normal distribution of
# errors
mean_deviation = 5

#--------------------------------------------------------------------------
# figure files
#--------------------------------------------------------------------------
fig_dir = '../Figures/MTJ'
os.makedirs(fig_dir, exist_ok=True)
shape_ratio_plot = r'../Figures/MTJ/shape_ratio'
stress_plot      = r'../Figures/MTJ/stress_directions'
P_T_plot         = r'../Figures/MTJ/P_T_axes'
Mohr_plot        = r'../Figures/MTJ/Mohr_circles'
faults_plot      = r'../Figures/MTJ/faults'
 
#--------------------------------------------------------------------------
# advanced control parameters (usually not needed to be changed)
#--------------------------------------------------------------------------
# number of iterations of the stress inversion 
N_iterations = 6

# number of initial stres inversions with random choice of faults
N_realizations = 10

# axis of the histogram of the shape ratio
shape_ratio_min = 0
shape_ratio_max = 1
shape_ratio_step = 0.025

shape_ratio_axis = np.arange(shape_ratio_min+0.0125, shape_ratio_max, shape_ratio_step)
 
# interval for friction values
friction_min  = 0.40
friction_max  = 1.00
friction_step = 0.05


#--------------------------------------------------------------------------
# create output directories if needed
all_files = (output_file, shape_ratio_plot, stress_plot, P_T_plot, Mohr_plot, faults_plot)
for f in all_files:
    folder = os.path.dirname(f)
    if not os.path.exists(folder):
        os.makedirs(folder)
