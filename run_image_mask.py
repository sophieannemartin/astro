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
import numpy as np
from scipy.optimize import curve_fit

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

mask = no_star4.flatten() < 3600
mask = mask > 3300
no_star4m = np.ma.masked_array(no_star4, mask).compressed()

hist, bin_edges = np.histogram(no_star4m,bins=300, range=(3300,3600))
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [32300, 3420, 10]

coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)

# Get the fitted curve
hist_fit = gauss(bin_centres, *coeff)

plt.figure()
plt.hist(no_star4m, bins=300,range=(3300,3600), label='Histogram')
plt.plot(bin_centres, hist_fit, label='Fitted data')
plt.legend()
plt.xlabel('Counts')
plt.ylabel('Number of pixels')
plt.show()

