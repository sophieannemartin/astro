# -*- coding: utf-8 -*-

from astropy.io import fits
import numpy.ma as ma
import matplotlib.pyplot as plt
import numpy as np
import functions.image_functions as funcs
from matplotlib.colors import LogNorm

hdulist = fits.open("/Users/sophie/Documents/Work/Year 3/Lab/Astro/two_circle_test.fits")
colorvalues = hdulist[0].data

def remove_background(data, bckg):
    no_bckg = data.copy()
    no_bckg[no_bckg > bckg] = 0
    return no_bckg

nb = remove_background(colorvalues, 239)
plt.figure()
plt.imshow(nb, norm=LogNorm(), origin='lower')
count, data = funcs.count_galaxies_variabler(nb) # Set the fixed r and background value
plt.show()