#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:07:46 2018

@author: sophie
"""


from astropy.io import fits
import functions.image_functions as image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

hdulist = fits.open('A1_mosaic_two.fits')
bleeding = hdulist[0].data
bleeding[bleeding==0] = 1

#count, catalog = image.count_galaxies_fixedr(bleeding, 6, 50)
count, catalog = image.count_galaxies_variabler(bleeding, 80)
