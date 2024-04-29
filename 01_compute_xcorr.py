#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:46:29 2024

@author: destin nziengui bâ (destin.nziengui-ba@univ-grenoble-alpes.fr / destin.nziengui@febus-optics.com)

Script for computing cross-correlation from reducted DAS data
"""


#
#   MODULE IMPORTATION 
#-------------------------


import os 
import time
import pandas
import numpy as np 
from compute_xcorr_utils import compute_xcorr_from_ReductedFile


#%%  


#
#      COUPLES
#-------------------------


# parameters 

source = 1125     # virtual source 
dx     = 2        # spatial sampling between channels 
L      = 200      # Desired length of the profile 
nrecs  = int(L/dx)  # number of virtual receivers 
receivers = source + -np.arange(nrecs) #virtual receivers

# Definition of the couples to correlate 
couples=[]
for rec in receivers :
        couples+=[[source,rec]]
        
couples=np.array(couples,dtype="int32")

#%%




#
#    NOISE PROCESSING
#-------------------------


recipes= ['white','taper'] 


# preprocessing methods  
arg_white = {'fmin'  : 0.5 ,'fmax'  : 25.  } #spectral whitening 
arg_taper = {'alpha' : 0.1}  #taper 


# Combine all the preprocessing methods
P_noise_processing = {}
for method in recipes:

    if 'arg_' + method in vars():
        #in_['args'].append(vars()['arg_' + method])
        P_noise_processing[method]=vars()['arg_' + method]


#%%
#
#    XCORR PARAMETERS
#----------------------------


# Define input parameters for cross-correlation computation
P_xcorr = {
    'cc_path_in': '/run/media/dnziengui/T7/CC3/example_margaux_UGA/data', # where the raw DAS data are stored 
    'cc_path_out': '/run/media/dnziengui/T7/CC3/example_margaux_UGA/xcorr', # where the cross correlations will be stored
    'cc_lag': 2.0,            # maximum lag time (seconds)
    'cc_WinSize': 5 * 60,     # window length (seconds)
    'cc_stack': True,         # Whether to stack or not (inside or outside)
    'cc_min_to_stack': 30 * 2,# number of minutes to stack and store
    'cc_save': True           # Whether to save or not 
}

# Create output directory if it doesn't exist
if not os.path.exists(P_xcorr['cc_path_out']):
    os.mkdir(P_xcorr['cc_path_out'])


# Save all the parameters for cross correlation computation in a single "pxc.npy" file 
P_out = {"couples": couples, "P_np": P_noise_processing, "P_xcorr": P_xcorr}
np.save(os.path.join(P_xcorr['cc_path_out'], "pxc.npy"), P_out)




#%%


#
#   XCORR COMPUTATION: LOOP OVER FILES
#--------------------------------------



# Define file containing list of files for cross-correlation computation
file_xcorr = os.path.join(P_xcorr['cc_path_in'],"liste_xcorr.txt")
files=pandas.read_csv(file_xcorr,header=None)
files=files.iloc[:,0].to_numpy(str)
nfiles=files.size



# Loop over the files 
print("Computing xcorr over %d files .... "%(nfiles))
for i in range(nfiles):

    start_time = time.time()
    filename=files[i]
    compute_xcorr_from_ReductedFile(filename,couples,P_noise_processing,P_xcorr,prefix="RSR")
    print("%s :--- %.2f seconds ---" % (filename,time.time() - start_time))


# around 45 sec/file 
