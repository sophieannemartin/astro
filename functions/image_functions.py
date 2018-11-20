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
    
def remove_background(data,mean):
    dataf = data.flatten()
    mphigh = ma.masked_where(dataf <= mean, dataf)
    no_background = np.reshape(mphigh, data.shape)
    return no_background

def remove_background2(data, mean):
    no_bckg = data.copy()
    no_bckg[no_bckg <= mean] = 0
    return no_bckg

    
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


def max_pixel(data):
    max_val = data.argmax()
    return max_val


def find_next_brightest(data):
    co_ords = np.unravel_index(np.argmax(data), data.shape)
    return co_ords


def make_circle(co_ordinates, data, r):
    # In the form y, x
    a,b = co_ordinates[1], co_ordinates[0]
    nx,ny = data.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y >= r*r
    tmp = np.copy(data)
    tmp[mask] = 1
    tmpcircle = np.ma.masked_array(tmp,mask)
    return tmpcircle

def bin_data(co_ordinates, data, r):
    a,b = co_ordinates[1], co_ordinates[0]
    nx,ny = data.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= r*r # get rid of current circle
    data[mask] = 1
    data = np.ma.masked_array(data,mask)
    return data

def locate_centre(data, x, y):
    left, right = scan_horizontal(data, x, y)
    mid = int((right-left)/2)
    x_mid = left+mid
    top, bottom = scan_vertical(data, x_mid, y)
    mid = int((top-bottom)/2)
    y_mid = bottom+mid
    return x_mid, y_mid

def count_galaxies(data):
    
    pos = find_next_brightest(data) # x and y are backwards
    brightest = data[pos[0], pos[1]]
    count = 0
    r = 1
    while brightest > 3450:

        pos = find_next_brightest(data)
        brightest = data[pos[0], pos[1]]
        yc, xc = locate_centre(data, pos[0], pos[1])
        search_area = make_circle((xc,yc), data, r)
        
        if ((len(search_area.compressed()))-np.count_nonzero(search_area.compressed())) != 0:
            print('object found')
            count+= 1
            data = bin_data(pos, data, r)
        
        else:
            print(search_area.compressed())
            r += 1
            continue
            
    return count

# Zooming out function before counting an object MAIN FUNCTION
def find_radius(data, xc, yc, brightness):
   """
   WORK ON THE UPDATING OF DATA WHEN BINNING SINCE BIN DATA 
   CREATES ZEROS IN THE TMP CIRCLE ARRAYY SINCE IT COPIES DATA?
   """
   r = 1
   tmp = np.copy(data)
   a,b = xc, yc
   nx,ny = data.shape
   y,x = np.ogrid[-a:nx-a,-b:ny-b]
   mask = x*x + y*y >= r*r
   tmp[mask] = 1
   tmpcircle = np.ma.masked_array(tmp,mask)
   
   # Define data inside the circle as a temp searching area
   while tmpcircle.__contains__(0) == False:
       r+=1
       tmp = np.copy(data)
       a,b = yc, xc
       nx,ny = data.shape
       y,x = np.ogrid[-a:nx-a,-b:ny-b]
       mask = x*x + y*y >= r*r
       tmp[mask] = 1
       tmpcircle = np.ma.masked_array(tmp,mask)
       
   return r
    
#def calc_magnitude(x,y,radius,localbkgd = bckg, ZPinst=ZPinst):
"DEFINE THE MAGNITUDES USING THEORY EQNS"

def count_galaxies_variabler(data):
    
    data_o = data
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(data, norm=LogNorm(), origin='lower')
    pos = find_next_brightest(data)
    brightest = data[pos[0], pos[1]]
    count = 0
    
    while brightest > 0:

        pos = find_next_brightest(data)
        brightest = data[pos[0], pos[1]]
        if brightest == 0:
            break
        else:
            yc, xc = locate_centre(data, pos[0], pos[1])
            r = find_radius(data_o, xc, yc, brightest)
            if r==1:
                c = plt.Circle((xc,yc), r, color='blue', fill=False)
                ax.add_artist(c)
                data = bin_data((xc, yc), data, r)
            else:
                c = plt.Circle((xc,yc), r, color='red', fill=False)
                ax.add_artist(c)
                data = bin_data((xc, yc), data, r)
                count+=1
            
    return count, data

    
def count_galaxies_fixedr(data, r, bckg):
    
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(data, norm=LogNorm(), origin='lower')
    pos = find_next_brightest(data)
    brightest = data[pos[0], pos[1]]
    count = 0
    
    while brightest > bckg:

        pos = find_next_brightest(data)
        brightest = data[pos[0], pos[1]]
        if brightest == 0:
            break
        else:
            yc, xc = locate_centre(data, pos[0], pos[1])
            c = plt.Circle((xc,yc), r, color='red', fill=False)
            ax.add_artist(c)
            data = bin_data((xc, yc), data, r)
            count+=1
        
    return count, data
        
 
def scan_horizontal(data, current_x, current_y):
    cursor_r = current_x
    cursor_l = current_x
    y = current_y
    
    while data[cursor_r,y] != 0:
        cursor_r += 1
    
    while data[cursor_l,y] != 0:
        cursor_l -= 1
        
    return cursor_r, cursor_l


def scan_vertical(data, current_x, current_y):
    cursor_u = current_y
    cursor_d = current_y
    x = current_x
    
    while data[x, cursor_u] != 0:
        cursor_u += 1
    
    while data[x, cursor_d] != 0:
        cursor_d -= 1
    
    return cursor_u, cursor_d
    
