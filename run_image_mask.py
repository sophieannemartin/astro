#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:10:58 2018

@author: sophie
"""
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from astropy.io import fits
import mask_image as image

hdulist = fits.open("/Users/sophie/Documents/Work/Year 3/Lab/Astro/A1_mosaic.fits")
pixelvalues = hdulist[0].data
pixels = pixelvalues.flatten()

no_edges = image.remove_edges(115, pixelvalues)
no_edgesf = no_edges.flatten()

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(pixelvalues, norm = LogNorm(), origin='lower')
plt.title('original image')

#mphigh = ma.masked_where(pixels >=6000, pixels, copy=True)
#plt.hist(mphigh.compressed(), 300, color = 'green', range=(3300,3600))


#removing vertical bleeding from stars
no_strip1 = image.remove_strip(1426,1447,0,4608,no_edges)
no_strip2 = image.remove_strip(772,779,3203,3417,no_strip1)
no_strip3 = image.remove_strip(970,978,2704,2835,no_strip2)
no_strip4 = image.remove_strip(901,908,2222,2357,no_strip3)
no_strip5 = image.remove_strip(2131,2137,3705,3805,no_strip4)

#removing bright stars using plt.circle
no_star1 = image.remove_star((3210,1432), 198, no_strip5)
no_star2 = image.remove_star((3322,774), 40 ,no_star1)
no_star3 = image.remove_star((2773,972),33, no_star2)
no_star4 = image.remove_star((2284,906), 29,no_star3)

#removing horizontal bleeding from main bleed from central star 
#no_strip5 = remove_strip(1102,1652,426,428,no_star)
#no_strip6 = remove_strip(,no_strip5)
    
plt.subplot(1,2,2)
plt.imshow(no_star4, norm = LogNorm(), origin = 'lower')
plt.title('edited image')

data_only  = image.remove_background(no_star4)

plt.figure(2)
plt.imshow(data_only, norm = LogNorm(), origin = 'lower')

'''
plt.figure(3)
plt.hist(pixels, 300, color = 'green', range = (3300,3600))
plt.hist(no_edgesf.compressed(), 300, color = 'blue', range = (3300,3600))
plt.xlabel('Counts')
plt.ylabel('Number of pixels')
plt.title('Histogram 300 bins')
'''
plt.figure(4)
plt.hist(data_only.compressed(), 10000, color = 'blue', 
         range=(min(data_only.compressed()), max(data_only.compressed())))

