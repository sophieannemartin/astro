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

plt.figure(0)
n = plt.hist(pixelsf, 300, color='#a42b44', range=(3300,3600), label='unmasked')
plt.xlabel('Counts')
plt.ylabel('Number of pixels')
plt.title('Histogram bins=300')


def remove_edges(width, data):
    
    """
    Removes edges assuming same width all around
    """
    
    mask = np.zeros(data.shape)
    mask[:width, :] = 1
    mask[-width:, :] = 1
    mask[:, :width] = 1
    mask[:, -width:] = 1

    data_noedges = np.ma.masked_array(data, mask)
    return data_noedges

pixels_noedges = remove_edges(115, pixels).flatten()

n = plt.hist(pixels_noedges.compressed(), 300, color='#ff9900', 
             range=(3300,3600), label='masked')
plt.xlabel('Counts')
plt.ylabel('Number of pixels')
plt.title('Histogram bins=300')
plt.legend()

