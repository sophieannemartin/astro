#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:50:27 2018

@author: sophie
"""

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from astropy.io import fits
import functions.image_functions as image
import numpy as np
from run_image_mask import no_background

galaxies = no_background.filled(0)
plt.imshow(galaxies, norm=LogNorm(), origin='lower')
