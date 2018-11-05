#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:49:09 2018

@author: annawilson
"""

import astropy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
from astropy.io import fits
import numpy.ma as ma

hdulist = fits.open("/Users/annawilson/Documents/University/Physics/Third_Year/Labs/Astro/A1_mosaic.fits")
pixelvalues = hdulist[0].data
pixels = pixelvalues.flatten()

plt.hist(pixels, 300, range = (3300,3600))

#mphigh = ma.masked_where(pixels >=6000, pixels, copy=True)

mppos = ma.masked_where(pixelvalues[0:115][], pixelvalues, copy=True)


plt.figure(1)

#plt.hist(mphigh.compressed(), 300, color = 'green', range=(3300,3600))
plt.hist(mppos.compressed(), 300, color = 'blue', range=(3300,3600))

plt.xlabel('Counts')
plt.ylabel('Number of pixels')
plt.title('Histogram 300 bins')

plt.show()
