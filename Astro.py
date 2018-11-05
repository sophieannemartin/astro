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


#mphigh = ma.masked_where(pixels >=6000, pixels, copy=True)
#plt.hist(mphigh.compressed(), 300, color = 'green', range=(3300,3600))

def remove_edges(width, data):
    
    """
    Removes edges of data with width
    """
    
    mask = np.zeros(data.shape)
    mask[:width,:] = 1
    mask[-width:,:] = 1
    mask[:,:width] = 1
    mask[:,-width:] = 1

    data_noedges = np.ma.masked_array(data, mask)
    return data_noedges

no_edges = remove_edges(115, pixelvalues)
no_edgesf = no_edges.flatten()


plt.figure(1)
plt.hist(pixels, 300, color = 'green', range=(20000,23000))
plt.hist(no_edgesf.compressed(), 300, color = 'blue', range=(20000,23000))

plt.xlabel('Counts')
plt.ylabel('Number of pixels')
plt.title('Histogram 300 bins')

plt.show()
