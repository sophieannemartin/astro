#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:53:23 2018

@author: sophie
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

path = '/Users/sophie/Desktop/Work/Year 3/Lab/Astro/A1_mosaic.fits'
pixels = fits.getdata(path)
pix_values = pixels.flatten()

#Remove extreme values out
pixelsf = pix_values[pix_values <= 3600]
pixelsf = pixelsf[pixelsf >= 3300]
             
# Plot histogram with data ignoring values where 3300 < value < 3600
plt.figure('Histogram')
plt.hist(pixelsf, 300, color='#a42b44', range=(3300,3600), label='unmasked')
plt.xlabel('Counts')
plt.ylabel('Number of pixels')
plt.title('Histogram bins=300')

def remove_edges(width, data):
    
    """
    Removes edges assuming same width all around, returns 2D array of data
    """
    
    mask = np.zeros(data.shape)
    mask[:width, :] = 1
    mask[-width:, :] = 1
    mask[:, :width] = 1
    mask[:, -width:] = 1

    data_noedges = np.ma.masked_array(data, mask)
    return data_noedges

# Remove edges and selects only non-masked data in array using .compressed
pixels_ne = remove_edges(115, pixels)
pixels_nef = pixels_ne.flatten().compressed()

plt.figure('Histogram')
plt.hist(pixels_nef, 300, color='#ff9900', range=(3300,3600), label='masked')
plt.xlabel('Counts')
plt.title('Histogram bins=300')
plt.legend()