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

galaxies = no_block3.filled(0)
plt.imshow(galaxies, norm=LogNorm(), origin='lower')

count, data, magnitudes = image.count_galaxies_fixedr(galaxies, 6, mu+5*sigma) 
# 4758 at r=30, used 6pixel radius from lab book