#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:39:02 2018

@author: sophie
"""

import numpy as np
import pandas as pd

def create_histogram_heights(catalog, range_, bins):
    # a dataframe
    hist_heights, edges = np.histogram(catalog['magnitude'].values, 
                               range=(range_[0], range_[1]), bins=bins)
    
    df = pd.DataFrame({'Bin_heights':hist_heights, 'Edges':edges[1:]})
    return df

def save_heights(df):
<<<<<<< HEAD
    writer = pd.ExcelWriter('histogram_of_magnitudes2.xlsx')
=======
    writer = pd.ExcelWriter('histogram_tmp.xlsx')
>>>>>>> 8320f18484dce0323038c7ccc47f71a7610a8fcc
    df.to_excel(writer)
    writer.save()
    
    
def save_log_data(filepath, range_, bins):
    catalog = pd.read_excel(filepath)
    df = create_histogram_heights(catalog, range_, bins)
    save_heights(df)
    return df

save_log_data('/Users/annawilson/Documents/GitHub/astro/catalog_stop_itself.xlsx', (8,20), 50)
