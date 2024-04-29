#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:46:29 2024

@author: destin nziengui bâ (destin.nziengui-ba@univ-grenoble-alpes.fr / destin.nziengui@febus-optics.com)

goal : load and plot cross correlations data 
"""


#
#   MODULE IMPORTATION 
#-------------------------

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from load_xcorr_utils import fiber_loc_file,load_xcorr



#
#       DATA LOADING
#-------------------------

# Define path to xcorr data
xcorr_data_path = '/run/media/dnziengui/T7/CC3/example_margaux_UGA/xcorr'

# Get list of xcorr files
files = np.sort(glob.glob(os.path.join(xcorr_data_path, "xcorr_*")))

# Load xcorr data from first two files
xcorr0, lags, distances, couples = load_xcorr(files, file_number=0)
xcorr1, _, _, _ = load_xcorr(files, file_number=1)


#
#         FIGURE 
#-------------------------


# Plot settings
amax = 0.6952247142791748
extent = [lags[0], lags[-1], distances[-1], distances[0]]
cmap = "seismic"
vmin, vmax = -amax, amax
aspect = "auto"
interpolation = "None"



# Plot fiber location
fig = plt.figure(figsize=(16, 9), constrained_layout=True)
ax = plt.subplot(221)
XY = fiber_loc_file()
plt.plot(*XY, "b.", label="channels")
plt.plot(*XY[:, couples[:, 0]], "r.", label="receivers")
plt.plot(*XY[:, couples[:, 1]], "g*", label="source")
plt.grid()
plt.gca().set_aspect("equal")
plt.legend()
plt.xlabel("Easting [m]")
plt.ylabel("Northing [m]")
plt.title("Fiber location")

# Plot xcorr data for first file
ax = plt.subplot(223)
s = ax.imshow(xcorr0, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation=interpolation, extent=extent)
plt.xlabel("Lag time [sec]")
plt.ylabel("Distance [meters]")
plt.title("datetime: 2022-05-06T14:06:49")

# Plot xcorr data for second file
ax = plt.subplot(224)
s = ax.imshow(xcorr1, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation=interpolation, extent=extent)
plt.xlabel("Lag time [sec]")
plt.colorbar(s, ax=ax)
plt.title("datetime: 2022-05-06T15:06:49")

plt.show()


fig.savefig("xcorr_linear_section.png")