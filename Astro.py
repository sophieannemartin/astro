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

hdulist = fits.open("/Users/annawilson/Documents/University/Physics/Third_Year/Labs/Astro/A1_mosaic.fits")
pixelvalues = hdulist[0].data
pixels = pixelvalues.flatten()

'''for i in range(5):
    plt.figure(i)
    plt.hist(pixelvalues[i], bins='auto', range=(3000,6000))
    plt.show()'''

plt.figure(1)
#n = plt.hist(pixels, bins=300, normed=True, range=(3300,3600))

# best fit of data
(mu, sigma) = norm.fit(pixels)

# the histogram of the data
n, bins, patches = plt.hist(pixels, 300, normed=True, facecolor='green', range=(3300,3600))

# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--')

#x=range(3300,3600)
#count = n[0]
#mean = np.mean(count)
#std_dev = np.sqrt(np.var(count))

#def gaussian(x, mu, sig):
#    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#plt.plot(x,gaussian(count,mean,std_dev))


plt.xlabel('Counts')
plt.ylabel('Number of pixels')
plt.title('Histogram 300 bins')
plt.show()
