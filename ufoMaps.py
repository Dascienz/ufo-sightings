#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:01:02 2017

@author: dascienz
"""
import os
cwd = os.getcwd()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.basemap import Basemap

def readData():
    data = pd.read_csv(cwd + "/ufoData.csv", index_col=0)
    return data

data = readData()
  
def worldMap(data):
    """Total world sightings map."""
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])

    m = Basemap(projection='kav7',lon_0 = 0,resolution = None)
    m.drawmapboundary(fill_color='0.3')
    m.shadedrelief()

    # Draw parallels and meridians.
    m.drawparallels(np.arange(-90.,99.,30.))
    m.drawmeridians(np.arange(-180.,180.,60.))

    # Calculate the point density
    lonx = data['lon'][np.isnan(data['lon']) == False].values
    laty = data['lat'][np.isnan(data['lat']) == False].values
    xy = np.vstack([lonx,laty])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = lonx[idx], laty[idx], z[idx]

    xmap, ymap = m(x,y)
    sc = m.scatter(xmap, ymap, c=z, s=20, edgecolor='', cmap=plt.cm.plasma)
    ax.set_title('World UFO Sightings')
    m.colorbar(sc, location='bottom', pad='3%')
    plt.savefig('World_UFO_Heatmap_300dpi.png', format='png', dpi = 300)
    plt.show()

def naMap(data):
    """North America sightings map."""
    fig = plt.figure()
    m = Basemap(projection='lcc',
                llcrnrlon=-119, 
                llcrnrlat=22, 
                urcrnrlon=-64,
                urcrnrlat=49, 
                lat_1=33, 
                lat_2=45,
                lon_0=-95,
                area_thresh=10000)

    m.drawmapboundary(linewidth= 1.5)
    m.drawstates()
    m.drawcountries()
    m.shadedrelief()

    # Calculate the point density
    lonx = data['lon'][np.isnan(data['lon']) == False].values
    laty = data['lat'][np.isnan(data['lat']) == False].values
    xy = np.vstack([lonx,laty])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = lonx[idx], laty[idx], z[idx]
    xmap, ymap = m(x, y)
    sc = m.scatter(xmap, ymap, c=z, s=30, edgecolor='', cmap=plt.cm.plasma)
    plt.title('United States UFO Sightings')
    m.colorbar(sc, location='right', pad='5%')
    plt.tight_layout()
    plt.savefig('America_UFO_Heatmap_300dpi.png', format='png', dpi = 300)
    plt.show()

"""
Is there a correlation with airport, heliport, and seaplane base location?
Airport data obtained and sliced from: https://www.faa.gov/airports/airport_safety/airportdata_5010/
"""
def readAirportData():
    airports = pd.read_csv('~/Desktop/PythonCode/Portfolio/UFO Sightings/airportData.csv')
    airports = airports.dropna().reset_index(drop=True)

    def convert(tude):
        multiplier = 1 if tude[-1] in ['N', 'E'] else -1
        return multiplier*sum(float(x)/60**n for n, x in enumerate(tude[:-1].split('-')))

    airports['Lon'] = [convert(l) for l in airports['ARPLongitude']]
    airports['Lat'] = [convert(l) for l in airports['ARPLatitude']]
    airports['ActivationDate'] = pd.to_datetime(airports['ActiviationDate'].astype('str'), errors = 'coerce')
    return airports

airports = readAirportData()

# Plotting airports for the USA
def airportMap(airports):
    fig = plt.figure()
    m = Basemap(projection='lcc',
                llcrnrlon=-119, 
                llcrnrlat=22, 
                urcrnrlon=-64,
                urcrnrlat=49, 
                lat_1=33, 
                lat_2=45,
                lon_0=-95,
                area_thresh=10000)

    m.drawmapboundary(linewidth= 1.5)
    m.drawstates()
    m.drawcountries()
    m.shadedrelief()

    # Calculate the point density
    lonx = airports['Lon'][(airports['ActivationDate'] >= pd.Timestamp('1950-01-01'))].values
    laty = airports['Lat'][(airports['ActivationDate'] >= pd.Timestamp('1950-01-01'))].values
    xy = np.vstack([lonx,laty])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = lonx[idx], laty[idx], z[idx]
    xmap, ymap = m(x, y)

    sc = m.scatter(xmap, ymap, c=z, s=30, edgecolor='', cmap=plt.cm.plasma)
    m.colorbar(sc, location='right', pad='5%')
    plt.title('United States Airports, Heliports, and Seaplane Bases')
    plt.tight_layout()
    plt.savefig('Airports.png', format='png', dpi = 300)
    plt.show()

"""
Is there any visible correlation between Sightings and Military Base locations?
Military bases obtained from 'https://www.google.com/maps/d/viewer?mid=1XFBnIuaJ-71hcaDJvdmBmeXNhYM&hl=en&ll=47.323248104393414%2C-114.05083400000001&z=3'
"""
def readMilitaryData():
    military = pd.read_csv(cwd + "/militaryData.csv")
    military = military.dropna().reset_index(drop=True)
    return military

military = readMilitaryData()

# Plotting Military Bases in the USA
def militaryMap(military):
    fig = plt.figure()
    m = Basemap(projection='lcc',
                llcrnrlon=-119, 
                llcrnrlat=22, 
                urcrnrlon=-64,
                urcrnrlat=49, 
                lat_1=33, 
                lat_2=45,
                lon_0=-95,
                area_thresh=10000)

    m.drawmapboundary(linewidth= 1.5)
    m.drawstates()
    m.drawcountries()
    m.shadedrelief()

    # Calculate the point density
    lonx = military['longitude'].values
    laty = military['latitude'].values
    xy = np.vstack([lonx,laty])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = lonx[idx], laty[idx], z[idx]
    xmap, ymap = m(x, y)
    
    sc = m.scatter(xmap, ymap, c=z, s=60, edgecolor='black', cmap=plt.cm.plasma)
    m.colorbar(sc, location='right', pad='5%')
    plt.title('United States Military Bases')
    plt.tight_layout()
    plt.savefig('military.png', format='png', dpi = 300)
    plt.show()