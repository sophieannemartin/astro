# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import astropy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
from astropy.io import fits
import numpy.ma as ma
from matplotlib.colors import LogNorm

hdulist = fits.open('\\\\icnas4.cc.ic.ac.uk\\aw2316\\Third_Year\\Labs\\Astro\\A1_mosaic.fits')
pixelvalues = hdulist[0].data
pixels = pixelvalues.flatten()

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
    
def remove_exp_bleeding1(x1,x2,y0,a,lamb,data):
    """
    Removes a section of data in exponential shape
    """
    mask = np.zeros(data.shape)         
    x = range(0,x2-x1)
    for i in x: 
        y = a*np.exp(i*lamb)
        y = int(round(y))
        mask[y0:y+y0,i+x1] = 1
    
    data_noexp = np.ma.masked_array(data,mask)
    return data_noexp

def remove_exp_bleeding2(x1,x2,y0,a,lamb,data):
    mask = np.zeros(data.shape)
    x = range(0,x1-x2,-1)
    for i in x:
        y = a*np.exp(i*lamb)
        y = int(round(y))
        mask[y0:y+y0,i+x2] = 1
    
    data_noexp = np.ma.masked_array(data,mask)
    return data_noexp
    
#removing edges of image  
no_edges = remove_edges(115, pixelvalues)

no_edgesf = no_edges.flatten()

#removing vertical bleeding from stars
no_strip1 = remove_strip(1426,1447,0,4608,no_edges)
no_strip2 = remove_strip(772,779,3203,3417,no_strip1)
no_strip3 = remove_strip(970,978,2704,2835,no_strip2)
no_strip4 = remove_strip(901,908,2222,2357,no_strip3)
no_strip5 = remove_strip(2131,2137,3705,3805,no_strip4)
no_strip6 = remove_strip(2130,2136,2280,2340,no_strip5)

#removing bright stars using plt.circle
no_star1 = remove_star((3210,1432), 198, no_strip6)
no_star2 = remove_star((3322,774), 40 ,no_star1)
no_star3 = remove_star((2773,972),33, no_star2)
no_star4 = remove_star((2284,906), 29,no_star3)
no_star5 = remove_star((3759,2133),28,no_star4)
no_star6 = remove_star((2312,2131), 26, no_star5)

no_star6f = no_star6.flatten()

#removing horizontal bleeding from main bleed from central star 
no_horiz1 = remove_exp_bleeding1(1447,1651,426,70,-0.018,no_star6)
no_horiz2 = remove_exp_bleeding2(1102,1429,426,70,0.01,no_horiz1)
no_horiz3 = remove_exp_bleeding1(1442,1702,313,92,-0.01,no_horiz2)
no_horiz4 = remove_exp_bleeding2(1019,1430,313,55,0.008,no_horiz3)
no_horiz5 = remove_strip(1392,1475,217,290,no_horiz4)

#plt.figure(1)
#plt.imshow(pixelvalues, norm = LogNorm(), origin='lower')
#plt.title('original image')

#plt.figure(2)
#plt.hist(pixels, 300, color = 'green', range = (3300,3600))
#plt.hist(no_star6f.compressed(), 300, color = 'blue', range = (3300,3600))
#plt.xlabel('Counts')
#plt.ylabel('Number of pixels')
#plt.title('Histogram 300 bins')

plt.figure(3)
plt.imshow(no_horiz5, norm = LogNorm(), origin = 'lower')
plt.title('edited image')

def remove_background(data,minn):
    """
    Masks data with values above or below those specified
    """
    dataf = data.flatten()
    mphigh2 = ma.masked_where(dataf <= minn, dataf)
    #mphigh = ma.masked_where(dataf >= maxx, dataf)
    detected = np.reshape(mphigh2, data.shape)
    return detected

no_background = remove_background(no_horiz2,3490)

#plt.figure(4)
#plt.imshow(no_background, norm = LogNorm(), origin = 'lower')
#plt.title('no background')

plt.show()