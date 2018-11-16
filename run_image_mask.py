#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:10:58 2018

@author: sophie
"""
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from astropy.io import fits
import functions.image_functions as image
import numpy as np
from scipy.optimize import curve_fit

hdulist = fits.open("A1_mosaic.fits")
pixelvalues = hdulist[0].data
pixels = pixelvalues.flatten()

no_edges = image.remove_edges(115, pixelvalues)
no_edgesf = no_edges.flatten()

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(pixelvalues, norm = LogNorm(), origin='lower')
plt.title('original image')

#removing vertical bleeding from stars
no_strip1 = image.remove_strip(1426,1447,0,4608,no_edges)
no_strip2 = image.remove_strip(772,779,3203,3417,no_strip1)
no_strip3 = image.remove_strip(970,978,2704,2835,no_strip2)
no_strip4 = image.remove_strip(901,908,2222,2357,no_strip3)
no_strip5 = image.remove_strip(2131,2137,3705,3805,no_strip4)
no_strip6 = image.remove_strip(2130,2136,2280,2340,no_strip5)

#removing bright stars using plt.circle
no_star1 = image.remove_star((3210,1432), 198, no_strip6)
no_star2 = image.remove_star((3322,774), 40 ,no_star1)
no_star3 = image.remove_star((2773,972),33, no_star2)
no_star4 = image.remove_star((2284,906), 29,no_star3)
no_star5 = image.remove_star((3759,2133),28,no_star4)
no_star6 = image.remove_star((2312,2131), 26, no_star5)

#removing horizontal bleeding from main bleed from central star 
no_horiz1 = image.remove_exp_bleeding1(1447,1651,426,70,-0.018,no_star6)
no_horiz2 = image.remove_exp_bleeding2(1102,1429,426,70,0.01,no_horiz1)
no_horiz3 = image.remove_exp_bleeding1(1442,1702,313,92,-0.01,no_horiz2)
no_horiz4 = image.remove_exp_bleeding2(1019,1430,313,55,0.008,no_horiz3)
no_horiz5 = image.remove_exp_bleeding1(1441,1476,231,59,-0.07,no_horiz4)
no_horiz6 = image.remove_exp_bleeding2(1390,1431,231,40,0.03,no_horiz5)
no_horiz7 = image.remove_exp_bleeding1(1439,1471,216,40,-0.07,no_horiz6)
no_horiz8 = image.remove_exp_bleeding2(1398,1429,216,40,0.08,no_horiz7)
no_horiz9 = image.remove_exp_bleeding1(1439,1524,123,60,-0.03,no_horiz8)
no_horiz10 = image.remove_exp_bleeding2(1290,1430,123,60,0.025,no_horiz9)
no_horiz11 = image.remove_strip(1390,1467,117,123,no_horiz10)

#removing misc blocks of saturation
no_block1 = image.remove_strip(1526,1538,117,139,no_horiz11)
no_block2 = image.remove_strip(1642,1647,334,354,no_block1)
no_block3 = image.remove_strip(1027,1042,424,451,no_block2)

plt.subplot(1,2,2)
plt.imshow(no_block3, norm = LogNorm(), origin = 'lower')
plt.title('edited image')

mask = no_block3.flatten() < 3600
mask = mask > 3300
no_block3m = np.ma.masked_array(no_block3, mask).compressed()

hist, bin_edges = np.histogram(no_block3m,bins=300, range=(3300,3600))
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [32300, 3420, 10]
coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
mu = coeff[1]
sigma = coeff[2]

# Get the fitted curve
hist_fit = gauss(bin_centres, *coeff)
plt.figure()
plt.hist(no_block3m, bins=300,range=(3300,3600), label='Histogram')
plt.plot(bin_centres, hist_fit, label='Fitted data')
plt.plot([mu+2*sigma,mu+2*sigma], [0, max(hist)], '--')
plt.legend()
plt.xlabel('Counts')
plt.ylabel('Number of pixels')