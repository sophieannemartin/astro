#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:11:14 2018

@author: sophie
"""

from astropy.io import fits
import functions.image_functions as image
import matplotlib.pyplot as plt

hdulist = fits.open('A1_mosaic_crop.fits')
test1 = hdulist[0].data
hdulistnb = fits.open('A1_mosaic_background.fits')
testnb = hdulistnb[0].data
hdulist2 = fits.open('A1_mosaic_two.fits')
test2 = hdulist2[0].data

global_background1 = 150 # Really important to define when its best to stop
global_backgroundnb = 250
global_background2 = 200

count, catalog = image.count_galaxies_fixedr(test1, 6, global_background1) 

countnb, catalognb = image.count_galaxies_fixedr(testnb, 6, global_backgroundnb) 

count2, catalog2 = image.count_galaxies_fixedr(test2, 12, global_background2) 

plt.show()