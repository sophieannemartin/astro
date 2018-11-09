#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:49:09 2018

@author: sophie
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
from astropy.io import fits
import numpy.ma as ma
from matplotlib.colors import LogNorm

hdulist = fits.open("/Users/sophie/Documents/Work/Year 3/Lab/Astro/A1_mosaic.fits")
pixelvalues = hdulist[0].data
pixels = pixelvalues.flatten()

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(pixelvalues, norm = LogNorm(), origin='lower')
plt.title('original image')

#mphigh = ma.masked_where(pixels >=6000, pixels, copy=True)
#plt.hist(mphigh.compressed(), 300, color = 'green', range=(3300,3600))

def remove_edges(width, data):
    
    """
    Removes edges of data with width 'width' from the edge of the image
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

def remove_strip(x1, x2, y1, y2, data):
    
    """
    Removes a strip of data values in block parameterized by x1, x2, y1 and y2
    """
    
    mask = np.zeros(data.shape)
    mask[y1:y2,x1:x2] = 1

    data_nostrip = np.ma.masked_array(data, mask)
    return data_nostrip
    
def remove_star(index,radius,data):
    
    """
    Removes a circle of data values parameterized by index (centre of circle) and radius
    """
    
    a,b = index #index takes the centre of the circle in the form (y,x)
    nx,ny = data.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= radius*radius
    data[mask] = 1

    data_nostar = np.ma.masked_array(data,mask)
    return data_nostar
    
    
#removing vertical bleeding from stars
no_strip1 = remove_strip(1426,1447,0,4608,no_edges)
no_strip2 = remove_strip(772,779,3203,3417,no_strip1)
no_strip3 = remove_strip(970,978,2704,2835,no_strip2)
no_strip4 = remove_strip(901,908,2222,2357,no_strip3)
no_strip5 = remove_strip(2131,2137,3705,3805,no_strip4)

#removing bright stars using plt.circle
no_star1 = remove_star((3210,1432), 198, no_strip5)
no_star2 = remove_star((3322,774), 40 ,no_star1)
no_star3 = remove_star((2773,972),33, no_star2)
no_star4 = remove_star((2284,906), 29,no_star3)

#removing horizontal bleeding from main bleed from central star 
#no_strip5 = remove_strip(1102,1652,426,428,no_star)
#no_strip6 = remove_strip(,no_strip5)
    
plt.subplot(1,2,2)
plt.imshow(no_star4, norm = LogNorm(), origin = 'lower')
plt.title('edited image')

'''
plt.figure(2)
plt.hist(pixels, 300, color = 'green', range = (3300,3600))
plt.hist(no_edgesf.compressed(), 300, color = 'blue', range = (3300,3600))
plt.xlabel('Counts')
plt.ylabel('Number of pixels')
plt.title('Histogram 300 bins')
'''


def detection(data):
    dataf = data.flatten()
    #mphigh = ma.masked_where(dataf >= 6000, dataf)
    mphigh2 = ma.masked_where(dataf <= 3490, dataf)
    detected = np.reshape(mphigh2, data.shape)
    return detected

detected = detection(no_star4)

plt.figure()
plt.imshow(detected, origin = 'lower')

plt.show()