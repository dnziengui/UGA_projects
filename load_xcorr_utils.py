#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:46:29 2024

@author: destin nziengui bâ (destin.nziengui-ba@univ-grenoble-alpes.fr / destin.nziengui@febus-optics.com)

utils to load xcorr files 
"""



import h5py
import numpy as np


def fiber_loc_file():
    
    """
    Load fiber locations from file.

    Returns:
        ndarray: Array containing X, Y coordinates of fiber channels.
    """
    
  
    # Open fiber position file
    h5file = h5py.File("/run/media/dnziengui/T7/CC3/example_margaux_UGA/data/location_2m.h5")
    # Extract fiber channel X, Y coordinates
    XY = h5file['/location/position'][0:2, :]  # extraction of X,Y coordinates
    h5file.close()  # Close the file
    return XY

def interstation_distances(couples):
    """
    Calculate distances between source and receiver fiber channels.

    Parameters:
        couples (ndarray): Array of fiber channel pairs.

    Returns:
        ndarray: Array of distances between source and receiver channels.
    """
    XY = fiber_loc_file()  # Load fiber locations
    sources = XY[:, couples[:, 0]]
    receivers = XY[:, couples[:, 1]]
    DX = (sources[0, :] - receivers[0, :])
    DY = (sources[1, :] - receivers[1, :])
    distances = np.sqrt(DX**2 + DY**2)  # Calculate distances
    return distances

def load_xcorr(files, file_number):
    """
    Load cross-correlation data from HDF5 file.

    Parameters:
        files (list): List of HDF5 files.
        file_number (int): Index of the file to load data from.

    Returns:
        ndarray: Cross-correlation data.
        ndarray: Array of lag times.
        ndarray: Array of distances between source and receiver channels.
        ndarray: Array of fiber channel pairs.
    """
    fname = files[file_number]  # Get filename
    f = h5py.File(fname, "r")  # Open HDF5 file
    xcorr = f["/xcorr"][0, :, :]  # Extract cross-correlation data
    couples = f["/ijx"][:]  # Extract fiber channel pairs
    lags = np.squeeze(f["/lags"][:])  # Extract lag times
    distances = interstation_distances(couples)  # Calculate distances
    f.close()  # Close the HDF5 file
    return xcorr, lags, distances, couples

