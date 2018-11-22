#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:50:27 2018

@author: sophie
"""

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import functions.image_functions as image
from run_image_mask import no_block3, mu, sigma
from astropy.io import fits

<<<<<<< HEAD
global_background = mu+6*sigma
catalogs = []

for s in range(2,6):
    global_background = mu+s*sigma
    galaxies = no_block3.filled(global_background) # background or 0 (what value should the masked bits be given)
    plt.imshow(galaxies, norm=LogNorm(), origin='lower')
    print('galaxy done s= %d' %s)
    count, catalog = image.count_galaxies_fixedr(galaxies, 6, global_background)
    catalogs.append(catalog)
# 4758 at r=30, used 6pixel radius from lab book
=======
global_background = mu+5*sigma
galaxies = no_block3.filled(global_background)
#plt.imshow(galaxies, norm=LogNorm(), origin='lower')

hdulist = fits.open('A1_mosaic_crop.fits')
test = hdulist[0].data

#count, catalog = image.count_galaxies_fixedr2(galaxies, 6, global_background) 
# 4758 at r=30, used 6pixel radius from lab book

count, catalog = image.count_galaxies_fixedr2(test, 6, global_background) 
>>>>>>> 6834312374dfbf4de608bfc5033447fc9b480343
