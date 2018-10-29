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
pixels = fits.getdata(path).flatten()


pixels_masked = pixels[pixels <= 3600]
pixels_masked = pixels_masked[pixels_masked >= 3300]
plt.figure(0)
n = plt.hist(pixels_masked, 300, color='purple')
plt.xlabel('Counts')
plt.ylabel('Number of pixels')
plt.title('Histogram bins=300')

