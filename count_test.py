# -*- coding: utf-8 -*-

from astropy.io import fits
import matplotlib.pyplot as plt

hdulist = fits.open("/Users/sophie/Documents/Work/Year 3/Lab/Astro/two_circle_test.fits")
colorvalues = hdulist[0].data
colors = colorvalues.flatten()


plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(colorvalues, norm = LogNorm(), origin='lower')
plt.title('test image')
