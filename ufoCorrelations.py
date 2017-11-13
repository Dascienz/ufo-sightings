#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:45:12 2017

@author: Dascienz
"""
import os
cwd = os.getcwd()

import numpy as np
import pandas as pd
from matplotlib import cm as cm
from matplotlib import pyplot as plt

def readData():
    data = pd.read_csv(cwd + "/ufoData.csv", index_col=0)
    data = data.dropna()
    data = data.reset_index(drop=True)
    return data

data = readData()
    
# Dictionary for changing month categories into number tags and vise versa.
months = {'Jan':1,'Feb':2,'Mar':3,
          'Apr':4,'May':5,'Jun':6,
          'Jul':7,'Aug':8,'Sep':9,
          'Oct':10,'Nov':11,'Dec':12}
inv_months = {v:k for k,v in months.items()}
data['month'] = data['month'].replace(inv_months)

# Correlation Matrix Function
duration = data['duration']
lat = data['lat']
lon = data['lon']
shape = pd.get_dummies(data['shape'])
color = pd.get_dummies(data['color'])
weekday = pd.get_dummies(data['weekday'])
month = pd.get_dummies(data['month'])
hour = pd.get_dummies(data['hour'])
year = pd.get_dummies(data['year'])
yearday = pd.get_dummies(data['yearday'])
model = pd.concat([duration,lat,lon,shape,color,weekday,month,hour,year,yearday],axis=1) 

def correlation_matrix(x):
    """Viewing correlation matrix if you'd like that sort of thing!"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax.imshow(x.corr(),interpolation='nearest', cmap=cmap)
    ax.grid(True)
    plt.axis('tight')
    plt.title('UFO Features Correlation Matrix')
    labels = x.columns.tolist()
    labels_dict = {labels[i]:i for i in range(len(labels))}
    plt.xticks(list(labels_dict.values()),list(labels_dict.keys()),rotation=45)
    plt.yticks(list(labels_dict.values()),list(labels_dict.keys()),rotation=45)
    # Add colorbar, specify tick locations to match desired ticklabels
    fig.colorbar(cax)
    plt.tight_layout(pad=0)
    #plt.savefig('Correlation_Matrix.png',format = 'png', dpi = 300)
    plt.show()  

def get_redundant_pairs(x):
    """Get diagonal and lower triangular pairs of correlation matrix."""
    pairs_to_drop = set()
    cols = x.columns
    for i in range(0, x.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(x, n=15):
    au_corr = x.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(x)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

get_top_abs_correlations(model, n=15)