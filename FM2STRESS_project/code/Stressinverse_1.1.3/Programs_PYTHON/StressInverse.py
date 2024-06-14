#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
-------------------------------------------------------------------------
                                                                         
   program STRESSINVERSE                                                 
                                                                         
   joint iterative inversion for stress and faults from focal mechanisms 
                                                                         
   Vavrycuk, V., 2014. Iterative joint inversion for stress and fault    
   orientations from focal mechanisms, Geophys. J. Int., 199, 69-77,     
   doi: 10.1093/gji/ggu224                                               
                                                                         
   version: 1.0                                                          
   last update : 03.07.2014                                              
                                                                         
   version: 1.1                                                          
   statistics toolbox is not further required                            
   output ASCII file with true strike, dip and rake angles is created    
   output ASCII file with principal focal mechanisms is created          
   last update : 07.11.2014                                              
                                                                         
   version: 1.1.1                                                        
   corrected function plot_mohr.py - selection of the half-plane          
   according to the principal faults                                     
   upper half-plane - first principal fault                              
   lower half-plane - second principal fault                             
   last update : 14.10.2018

   version: 1.1.2                                                        
   corrected function stability_criterion.m - sometimes the eigenvectors 
   of stress tensor were not correctly associated to corresponding       
   eigenvalues                                                           
   last update : 2.12.2019                                               
                                                                         
   version: 1.1.3                                                        
   function slip_deviation.py is included                                 
   the function calculates theoretical and predicted deviations of slip  
   on faults for the optimum stress                                                            
   last update : 23.11.2020
                                                                         
   copyright:                                                            
   The rights to the software are owned by V. Vavrycuk. The code can be  
   freely used for research purposes. The use of the software for        
   commercial purposes with no commercial licence is prohibited. 
"""

# Magics for ipython shell
try:
    from IPython import get_ipython
    ipython = get_ipython()
    # reset variables
    ipython.magic('reset -sf')
    ipython = get_ipython()
    # automatically reload files if they were edited and saved during ipython session
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
except Exception as e:
    pass

# Load a backend for plotting to png files
import matplotlib
matplotlib.use('Agg') 

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# reading input parameters
# ------------------------------------------------
import Input_parameters as ip

# ------------------------------------------------
# reading input data
# ------------------------------------------------
# focal mechanism
import read_mechanism as rm
strike_orig_1, dip_orig_1, rake_orig_1, strike_orig_2, dip_orig_2, rake_orig_2 = rm.read_mechanisms(ip.input_file)

## solution from noise-free data
# ------------------------------------------------
# inversion for stress
# ------------------------------------------------
import stress_inversion as si
tau_optimum,shape_ratio,strike,dip,rake,instability,friction = si.stress_inversion(
    strike_orig_1,dip_orig_1,rake_orig_1,strike_orig_2,dip_orig_2,rake_orig_2,
    ip.friction_min,ip.friction_max,ip.friction_step,ip.N_iterations, ip.N_realizations)     # inverze napeti z ohniskovych mechanismu, Michael (1984,1987)

# ------------------------------------------------
# optimum principal stress axes
# ------------------------------------------------
import azimuth_plunge as ap
diag_tensor, vector = np.linalg.eig(tau_optimum)

value = [diag_tensor[0], diag_tensor[1], diag_tensor[2]]
value_sorted = np.sort(value)
j = np.argsort(value)

sigma_vector_1_optimum  = np.array(vector[:,j[0]])
sigma_vector_2_optimum  = np.array(vector[:,j[1]])
sigma_vector_3_optimum  = np.array(vector[:,j[2]])

direction_sigma_1, direction_sigma_2, direction_sigma_3 = ap.azimuth_plunge(tau_optimum)

#--------------------------------------------------------------------------
# slip deviations
#--------------------------------------------------------------------------
import slip_deviation as sd
slip_deviation_1, slip_deviation_2 = sd.slip_deviation(tau_optimum,strike,dip,rake)

# ------------------------------------------------
# principal focal mechanism
# ------------------------------------------------
import principal_mechanisms as pm
principal_strike, principal_dip, principal_rake = pm.principal_mechanisms(sigma_vector_1_optimum,sigma_vector_3_optimum,friction)

## solutions from noisy data
# ------------------------------------------------
# loop over noise realizations
# ------------------------------------------------
import noisy_mechanisms as nm
import statistics_stress_inversion as ssi

n_error_ = np.zeros(ip.N_noise_realizations)
u_error_ = np.zeros(ip.N_noise_realizations)

sigma_vector_1_statistics = np.zeros((3,ip.N_noise_realizations))
sigma_vector_2_statistics = np.zeros((3,ip.N_noise_realizations))
sigma_vector_3_statistics = np.zeros((3,ip.N_noise_realizations))
shape_ratio_statistics = np.zeros(ip.N_noise_realizations)

for i in range(ip.N_noise_realizations):
    
    # superposition of noise to focal mechanisms
    strike1,dip1,rake1,strike2,dip2,rake2,n_error,u_error = nm.noisy_mechanisms(ip.mean_deviation,strike_orig_1,dip_orig_1,rake_orig_1)    
    
    n_error_[i] = np.mean(n_error)
    u_error_[i] = np.mean(u_error)
    
    sigma_vector_1,sigma_vector_2,sigma_vector_3,shape_ratio_noisy = ssi.statistics_stress_inversion(strike1,dip1,rake1,strike2,dip2,rake2,friction,ip.N_iterations,ip.N_realizations)
    
    sigma_vector_1_statistics [:,i] = sigma_vector_1
    sigma_vector_2_statistics [:,i] = sigma_vector_2
    sigma_vector_3_statistics [:,i] = sigma_vector_3
    shape_ratio_statistics    [i]   = shape_ratio_noisy
    
# ------------------------------------------------
# calculation of errors of the stress inversion
# ------------------------------------------------

sigma_1_error_statistics = np.zeros(ip.N_noise_realizations)
sigma_2_error_statistics = np.zeros(ip.N_noise_realizations)
sigma_3_error_statistics = np.zeros(ip.N_noise_realizations)    
shape_ratio_error_statistics = np.zeros(ip.N_noise_realizations)

for i in range(ip.N_noise_realizations):
    

    sigma_1_error_statistics[i] = np.real(np.arccos(np.abs(np.dot(sigma_vector_1_statistics[:,i],sigma_vector_1_optimum)))*180/np.pi)
    sigma_2_error_statistics[i] = np.real(np.arccos(np.abs(np.dot(sigma_vector_2_statistics[:,i],sigma_vector_2_optimum)))*180/np.pi)
    sigma_3_error_statistics[i] = np.real(np.arccos(np.abs(np.dot(sigma_vector_3_statistics[:,i],sigma_vector_3_optimum)))*180/np.pi)

    shape_ratio_error_statistics[i] = 100 * np.abs((shape_ratio-shape_ratio_statistics[i])/shape_ratio)

# ------------------------------------------------
# confidence limits
# ------------------------------------------------

mean_n_error = np.mean(n_error)
mean_u_error = np.mean(u_error)

max_sigma_1_error = np.max(sigma_1_error_statistics)
max_sigma_2_error = np.max(sigma_2_error_statistics)
max_sigma_3_error = np.max(sigma_3_error_statistics)

max_shape_ratio_error = np.max(np.abs(shape_ratio_error_statistics))

# ------------------------------------------------
# saving the results
# ------------------------------------------------
import scipy.io as sio

sigma_1 = {'azimuth': direction_sigma_1[0], 'plunge': direction_sigma_1[1], }
sigma_2 = {'azimuth': direction_sigma_2[0], 'plunge': direction_sigma_2[1], }
sigma_3 = {'azimuth': direction_sigma_3[0], 'plunge': direction_sigma_3[1], }


sigma_1_data = np.transpose(np.array([direction_sigma_1[0], direction_sigma_1[1]]))
sigma_2_data = np.transpose(np.array([direction_sigma_2[0], direction_sigma_2[1]]))
sigma_3_data = np.transpose(np.array([direction_sigma_3[0], direction_sigma_3[1]]))

mechanisms = {'strike': strike, 'dip': dip, 'rake': rake, }

mechanisms_data = np.transpose(np.array([strike, dip, rake]))
principal_mechanisms_data = np.transpose(np.array([principal_strike, principal_dip, principal_rake]))

principal_mechanisms = {'strike': principal_strike, 'dip': principal_dip, 'rake': principal_rake, }

data_matlab = {
        'sigma_1': sigma_1,
        'sigma_2': sigma_2,
        'sigma_3': sigma_3,
        'shape_ratio': shape_ratio,
        'mechanisms': mechanisms,
        'mechanisms_data': mechanisms_data,
        'friction': friction,
        'principal_mechanisms': principal_mechanisms,
        }

# matlab file with dict data type
sio.savemat(ip.output_file + ".mat", data_matlab)

# .dat files with array-like data
np.savetxt(ip.output_file + ".dat", mechanisms_data,  fmt='%1.4e')
np.savetxt(ip.principal_mechanisms_file + ".dat", principal_mechanisms_data, fmt='%1.4e')

## ------------------------------------------------
## plotting the results
## ------------------------------------------------
# -------------------------------------------------
# P/T axes and the optimum principal stress axes
# -------------------------------------------------
import plot_stress as plots
plots.plot_stress(tau_optimum,strike,dip,rake,ip.P_T_plot)

# ------------------------------------------------
# Mohr cicle diagram
# ------------------------------------------------
import plot_mohr as plotm
plotm.plot_mohr(tau_optimum,strike,dip,rake,principal_strike,principal_dip,principal_rake,ip.Mohr_plot)

# ------------------------------------------------
# confidence limiuts of the principal stress axes
# ------------------------------------------------
import plot_stress_axes as plotsa
plotsa.plot_stress_axes(sigma_vector_1_statistics,sigma_vector_2_statistics,sigma_vector_3_statistics,ip.stress_plot)

# ------------------------------------------------
# confidence limits (histogram) of the shape ratio
# ------------------------------------------------

pltHist, axH = plt.subplots()
n, bins, patches = axH.hist(x = shape_ratio_statistics, bins = ip.shape_ratio_axis, color='#0504aa', alpha = 0.7, rwidth=0.85)
axH.set_title('Shape ratio', fontsize = 14)
axH.grid(True)

# saving the plot
pltHist.savefig(ip.shape_ratio_plot + '.png')
plt.close(pltHist)
