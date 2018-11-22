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
    
    df = pd.DataFrame({'Bin_heights':hist_heights, 'Edges':edges})
    return df

def save_heights(df):
    
    writer = pd.ExcelWriter('histogram_of_magnitudes.xlsx')
    df.to_excel(writer)
    writer.save()
    
    
def save_log_data(catalog, range_, bins):
    df = create_histogram_heights
    save_heights(df)
    return df
