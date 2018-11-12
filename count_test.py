# -*- coding: utf-8 -*-

from astropy.io import fits
import numpy.ma as ma
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

hdulist = fits.open("/Users/sophie/Documents/Work/Year 3/Lab/Astro/two_circle_test.fits")
colorvalues = hdulist[0].data
colors = colorvalues.flatten()

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(colorvalues, norm = LogNorm(), origin='lower')
plt.title('test image')

def count_bybrightness(data):
    
    log = {}
    count = 0
    brightest = max(data.compressed())
    
    dataf = data.flatten()
 
    star = ma.masked_where(dataf == brightest, dataf)
    bright = np.reshape(star, data.shape)
    bright = np.array(bright, dtype=np.float64)
    return bright

star_bright = count_bybrightness(colors_nobackground)
plt.figure(2)
plt.imshow(star_bright, norm = LogNorm(), origin='lower')
plt.show()
