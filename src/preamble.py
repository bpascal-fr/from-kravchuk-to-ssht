# LOAD NECESSARY PYTHON LIBRARIES

import sys

sys.path.append("../src")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import random
import cmocean

# IMPORT USEFUL FUNCTIONS

from scipy.io import wavfile


# LOAD THE FUNCTIONS OF THE KRAVCHUK TOOLBOX

from elementary_signals import the_noisy_chirp, the_noisy_dirac, the_noisy_sine, the_coherent_state, the_noisy_coherent_state
from elementary_signals import display_signal, time_and_precision, the_white_noise
from pkravchuk_transform import the_ptransform
from ssht_transform import the_ssht_transform, the_new_transform, the_inverse_transform, rotate_signal, the_spherical_angles
from kravchuk_display import planar_display, spherical_display
from stft_transform import the_stft_transform, stft_display


# HANDLE APPEARANCE OF THE PLOTS

from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20
mpl.rcParams["axes.titlesize"] = 20
plt.rc("axes", labelsize=22.5)
plt.rc("legend", fontsize=20)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
mpl.rcParams["font.family"] = "roman"


# DEFINE THE COLORS FOR THE PLOTS
my_cmap   = cmocean.cm.cmap_d['deep']
c         = np.zeros((5,3))
rgba      = my_cmap(0.999)
c[0,:]    = rgba[:3]
rgba      = my_cmap(0.8)
c[1,:]    = rgba[:3]
rgba      = my_cmap(0.6)
c[2,:]    = rgba[:3]
rgba      = my_cmap(0.4)
c[3,:]    = rgba[:3]
rgba      = my_cmap(0.2)
c[4,:]    = rgba[:3]
rgba      = my_cmap(0.85)
cm        = rgba[:3]
d         = np.zeros((7,3))
d[0,:]    = np.array([0,0,0])
d[1:-1,:] = c
rgba      = my_cmap(0.1)
d[-1,:]   = rgba[:3]

