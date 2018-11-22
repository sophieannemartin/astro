#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:49:09 2018
Galaxy detection function definitions
@author: sophie
"""

import numpy as np
import numpy.ma as ma
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits

# FUNCTIONS THAT GET THE DATA ITSELF
# -----------------------------------
def get_pixel_values(file="A1_mosaic.fits"):
    hdulist = fits.open(file)
    pixelvalues = hdulist[0].data
    return pixelvalues

def get_zpinst(file="A1_mosaic.fits"):
    hdulist = fits.open(file)
    zpinst = hdulist[0].header['MAGZPT']
    zperr = hdulist[0].header['MAGZRR']
    return zpinst, zperr

# FUNCTIONS THAT MASK THE DATA
# ------------------------------
def remove_edges(width, data):
    
    """
    Removes edges of data with 'width' from the edge of the image
    """
    
    mask = np.zeros(data.shape)
    mask[:width,:] = 1
    mask[-width:,:] = 1
    mask[:,:width] = 1
    mask[:,-width:] = 1

    data_noedges = np.ma.masked_array(data, mask)
    return data_noedges


def remove_strip(x1, x2, y1, y2, data):
    
    """
    Removes a strip of data values in block parameterized by x1, x2, y1 and y2
    """
    
    mask = np.zeros(data.shape)
    mask[y1:y2,x1:x2] = 1

    data_nostrip = np.ma.masked_array(data, mask)
    return data_nostrip
    
def remove_star(index,radius,data):
    
    """
    Removes a circle of data values parameterized by index (centre of circle) and radius
    """
    
    a,b = index #index takes the centre of the circle in the form (y,x)
    nx,ny = data.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= radius*radius
    data[mask] = 1

    data_nostar = np.ma.masked_array(data,mask)
    return data_nostar
    

def remove_background(data, global_background):
    """
    Function to remove the background values and only leave the galaxy data
    """
    # Returns a masked array
    dataf = data.flatten()
    mphigh = ma.masked_where(dataf <= global_background, dataf)
    no_background = np.reshape(mphigh, data.shape)
    return no_background
    

def remove_exp_bleeding1(x1,x2,y0,a,lamb,data):
    """
    Removes a section of data in exponential shape
    """
    mask = np.zeros(data.shape)         
    x = range(0,x2-x1)
    for i in x: 
        y = a*np.exp(i*lamb)
        y = int(round(y))
        mask[y0:y+y0,i+x1] = 1
    
    data_noexp = np.ma.masked_array(data,mask)
    return data_noexp


def remove_exp_bleeding2(x1,x2,y0,a,lamb,data):
    mask = np.zeros(data.shape)
    x = range(0,x1-x2,-1)
    for i in x:
        y = a*np.exp(i*lamb)
        y = int(round(y))
        mask[y0:y+y0,i+x2] = 1
    
    data_noexp = np.ma.masked_array(data,mask)
    return data_noexp

# FUNCTIONS TO APPLY TO THE DATA ITSELF
# -------------------------------------

def find_next_brightest(data):
    """
    Returns the co-ordinates of next brightest pixel
    """
    co_ords = np.unravel_index(np.argmax(data), data.shape)
    return co_ords


def bin_data(co_ordinates, data, r):
    """
    Removes the data within a circle of radius r - go to 0
    Returns a masked array
    """
    a,b = co_ordinates[1], co_ordinates[0]
    nx,ny = data.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= r*r # gets rid of current circle sets to null atm
    data[mask] = 1
    data_n = np.ma.masked_array(data,mask)
    return data_n


def scan_horizontal(data, current_x, current_y, background):
    """
    Moves horizontally until the value == the background value
    Returns the co_ordinates
    Note: arrays work as rows by columns therefore x is the second element
    """
    cursor_r = current_x
    cursor_l = current_x
    y = current_y
    
    while data[y, cursor_r] > background:
        if cursor_r < data.shape[1]-1 and cursor_r>0:
            cursor_r += 1
        else:
            break
    
    while data[y, cursor_l] > background and cursor_l>0:
        if cursor_l < data.shape[1]-1:
            cursor_l -= 1
        else:
            break
        
    return cursor_r, cursor_l


def scan_vertical(data, current_x, current_y, background):
    """
    Moves vertically until the value == the background value
    Returns the co_ordinates
    Note: arrays work as rows by columns therefore x is the second element
    """
    cursor_u = current_y
    cursor_d = current_y
    x = current_x
    
    while data[cursor_u, x] > background and cursor_u>0:
        if cursor_u < data.shape[0]-1:
            cursor_u += 1
        else:
            break
    
    while data[cursor_d, x] > background and cursor_d>0:
        if cursor_d < data.shape[0]-1:
            cursor_d -= 1
        else:
            break
    
    return cursor_u, cursor_d
    

def locate_centre(data, x, y, bckg=0): # zero default
    """
    Locates the centre using the mid point of the scanning results
    """
    left, right = scan_horizontal(data, x, y, bckg)
    mid = int((right-left)/2)
    x_mid = left+mid
    top, bottom = scan_vertical(data, x_mid, y, bckg)
    mid = int((top-bottom)/2)
    y_mid = bottom+mid
    return x_mid, y_mid


def find_radius(data, xc, yc, bckg):
    
   # Zooming out function before counting an object
   """
   MUST WORK ON THE UPDATING OF DATA WHEN BINNING SINCE BIN DATA 
   CREATES ZEROS IN THE TMP CIRCLE ARRAYY SINCE IT COPIES DATA?
   """
   r = 1
   tmp = np.copy(data)
   a,b = xc, yc
   nx,ny = data.shape
   y,x = np.ogrid[-a:nx-a,-b:ny-b] # In form y, x due to the column notation in python
   mask = x*x + y*y >= r*r
   tmp[mask] = bckg
   tmp = np.ma.masked_array(tmp,mask).filled(bckg)

   # Define data inside the circle as a temp searching area
   while np.amin(tmp) >= bckg:
       r+=1
       tmp = np.copy(data)
       a,b = yc, xc
       nx,ny = data.shape
       y,x = np.ogrid[-a:nx-a,-b:ny-b]
       mask = x*x + y*y >= r*r # mask everything outside with bckg
       tmp[mask] = bckg
       tmp = np.ma.masked_array(tmp,mask).filled(bckg)
   return r

def scan_radius(data, xc, yc, bckg):
    
   # Zooming out function before counting an object
   """
   MUST WORK ON THE UPDATING OF DATA WHEN BINNING SINCE BIN DATA 
   CREATES ZEROS IN THE TMP CIRCLE ARRAYY SINCE IT COPIES DATA?
   """
   
   radii = []
   
   left, right = scan_horizontal(data, xc, yc, bckg)
   up, down = scan_vertical(data, xc, yc, bckg)
   dist_left = xc-left
   dist_right = right-xc
   dist_up = up-yc
   dist_down = yc-down
   radii.append(dist_left)
   radii.append(dist_right)
   radii.append(dist_up)
   radii.append(dist_down)
   
   radius = max(radii)
   return radius
   
   
def annular_ref(data, co_ords, r):
    # do not change the data
    tmp = data.copy()
    inner_r = 2*r
    outer_r = 4*r#twice the radius
    a,b = co_ords[1], co_ords[0]
    nx,ny = tmp.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask1 = x*x + y*y <= inner_r*inner_r # get rid of data inside current circle
    mask2 = x*x + y*y >= outer_r*outer_r # get rid of data outside annulus
    tmp[mask1] = 1
    tmp[mask2] = 1
    annulus = np.ma.masked_array(tmp,mask1).filled(0)
    annulus = np.ma.masked_array(annulus, mask2).filled(0)
    # Find mean of the values of the data left
    mean_bckg = np.mean(np.array(np.nonzero(annulus)))
    return mean_bckg


def annular_ref_vary(data, co_ords, r):
    # do not change the data
    tmp = data.copy()
    outer_r = r+(0.5*r) #1/2r + the radius
    a,b = co_ords[1], co_ords[0]
    nx,ny = tmp.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask1 = x*x + y*y <= r*r # get rid of data inside current circle
    mask2 = x*x + y*y >= outer_r*outer_r # get rid of data outside annulus
    tmp[mask1] = 1
    tmp[mask2] = 1
    annulus = np.ma.masked_array(tmp,mask1).filled(0)
    annulus = np.ma.masked_array(annulus, mask2).filled(0)
    # Find mean of the values of the data left
    mean_bckg = np.mean(np.array(np.nonzero(annulus)))
    return mean_bckg
    
    
def find_magnitude(data, co_ords, r, zpinst):
    # Do not alter the data yet!
    tmp = data.copy()
    # Look at certain part of the data
    a,b = co_ords[1], co_ords[0]
    nx,ny = tmp.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y >= r*r # get rid of data outside current circle
    tmp = np.ma.masked_array(tmp,mask).filled(0)
    # Count the values of the data left
    total_counts = np.sum(tmp)
    total_pix = np.count_nonzero(tmp)
    mean_bkg  = annular_ref(data, co_ords, r)
    source_counts = total_counts-(mean_bkg*total_pix) # subtracting the background
    mag = -2.5*np.log10(source_counts)
    if source_counts <0:
        print('oops')
    m = zpinst + mag
    return m, total_pix, mean_bkg

def find_magnitude_vary(data, co_ords, r, zpinst):
    # Do not alter the data yet!
    tmp = data.copy()
    # Look at certain part of the data
    a,b = co_ords[1], co_ords[0]
    nx,ny = tmp.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y >= r*r # get rid of data outside current circle
    tmp = np.ma.masked_array(tmp,mask).filled(0)
    # Count the values of the data left
    total_counts = np.sum(tmp)
    total_pix = np.count_nonzero(tmp)
    mean_bkg  = annular_ref_vary(data, co_ords, r)
    source_counts = total_counts-(mean_bkg*total_pix) # subtracting the background
    mag = -2.5*np.log10(source_counts)
    if source_counts <0:
        print('oops')
    m = zpinst + mag
    return m, total_pix, mean_bkg


def count_galaxies_variabler(data, bckg):
    
    data_o = data
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(data, norm=LogNorm(), origin='lower')
    pos = find_next_brightest(data)
    brightest = data[pos]
    count = 0
    local_backgrounds = [bckg]
    zpinst, zperr = get_zpinst()
    
    while brightest > bckg and brightest > local_backgrounds[-1]:

        pos = find_next_brightest(data)
        brightest = data[pos]
        if brightest <= bckg:
            break
        else:
            xc, yc = pos[1], pos[0]
            r = scan_radius(data_o, xc, yc, bckg)
            if r==1:
                c = plt.Circle((xc,yc), r, color='blue', fill=False)
                ax.add_artist(c)
                data = bin_data((xc, yc), data, r)
            else:
                mag, numberofpix, local_bckg = find_magnitude_vary(data, (xc,yc), r, zpinst)
                c = plt.Circle((xc,yc), r, color='red', fill=False)
                ax.add_artist(c)
                data = bin_data((xc, yc), data, r)
                count+=1
            
    return count, data


def count_galaxies_fixedr_stop(data, r, bckg): #uses a global background
    
    # does not need to do the locate centre
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(data, norm=LogNorm(), origin='lower')
    pos = find_next_brightest(data)
    brightest = data[pos[0], pos[1]]
    count = 0
    catalog = pd.DataFrame(columns=['x', 'y', 'magnitude', 'total counts', 
                                    'error', 'background', 'aperture_size'])
    zpinst, zperr = get_zpinst()
    local_backgrounds = [bckg]
    
    while brightest > bckg and brightest > local_backgrounds[-1]:
        print(brightest, local_backgrounds[-1])
        pos = find_next_brightest(data)
        yc, xc = pos[0], pos[1]
        brightest = data[pos]
        if brightest <= bckg:
            break
        else:
            mag, numberofpix, local_bckg= find_magnitude(data, (xc,yc), r, zpinst)
            local_backgrounds.append(local_bckg)
            c = plt.Circle((xc,yc), r, color='red', fill=False)
            ax.add_artist(c)
            data = bin_data((xc, yc), data, r)
            pos = find_next_brightest(data)
            count+=1
            catalog = catalog.append({'x':xc, 'y':yc, 'magnitude':mag, 
                            'total counts':numberofpix, 'error': zperr,
                            'background':local_bckg, 
                            'aperture_size':r}, ignore_index=True)        
    return count, catalog

def count_galaxies_fixedr(data, r, bckg): #uses a global background
    
    # does not need to do the locate centre
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(data, norm=LogNorm(), origin='lower')
    pos = find_next_brightest(data)
    brightest = data[pos[0], pos[1]]
    count = 0
    catalog = pd.DataFrame(columns=['x', 'y', 'magnitude', 'total counts', 
                                    'error', 'background', 'aperture_size'])
    zpinst, zperr = get_zpinst()
    
    while brightest > bckg:
        pos = find_next_brightest(data)
        yc, xc = pos[0], pos[1]
        brightest = data[pos]
        if brightest <= bckg:
            break
        else:
            mag, numberofpix, local_bckg= find_magnitude(data, (xc,yc), r, zpinst)
            c = plt.Circle((xc,yc), r, color='red', fill=False)
            ax.add_artist(c)
            data = bin_data((xc, yc), data, r)
            pos = find_next_brightest(data)
            count+=1
            catalog = catalog.append({'x':xc, 'y':yc, 'magnitude':mag, 
                            'total counts':numberofpix, 'error': zperr,
                            'background':local_bckg, 
                            'aperture_size':r}, ignore_index=True)        
    return count, catalog
  
    