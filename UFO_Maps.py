#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:01:02 2017

@author: dascienz
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

reports = pd.read_csv('~/Desktop/Python Code/Big Data Code/FinalProject/UFO_Dataset.csv')
del reports['Unnamed: 0']
reports = reports.reset_index(drop = True)  

# Dictionary for changing weekday categories into number tags.
weekdays = {'Monday':1,'Tuesday':2,'Wednesday':3,
        'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
        
# Vise versa
inv_weekdays = {v: k for k, v in weekdays.iteritems()}

# Dictionary for changing shape categories into number tags.
shape_labels = sorted(reports.Shape.value_counts().index.tolist())
shapes = {shape_labels[i]:i for i in xrange(len(shape_labels))}

# Vise versa
inv_shapes = {v: k for k, v in shapes.iteritems()}

# Dictionary for changing month categories into number tags and vise versa.
months = {'Jan':1,'Feb':2,'Mar':3,
          'Apr':4,'May':5,'Jun':6,
          'Jul':7,'Aug':8,'Sep':9,
          'Oct':10,'Nov':11,'Dec':12}
        
# Vise versa
inv_months = {v: k for k, v in months.iteritems()}
              
# Dictionary for changing color categories into number tags and vise versa.
color_labels = sorted(reports.Color.value_counts().index.tolist())
colors = {color_labels[i]:i for i in xrange(len(color_labels))}

# Vise versa
inv_colors = {v: k for k, v in colors.iteritems()}          
          
model = pd.read_csv('~/Desktop/Python Code/Big Data Code/FinalProject/UFO_Modeling.csv')
del model['Unnamed: 0']
model = model.reset_index(drop = True)

data = model
data = data.dropna().reset_index(drop = True)
data = data.replace({"Weekday": inv_weekdays,
                     "Shape": inv_shapes,
                     "Month": inv_months,
                     "Color": inv_colors})                    
  
# TOTAL WORLD REPORTS MAP Heatmap
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])

m = Basemap(projection='kav7',lon_0 = 0,resolution = None)
m.drawmapboundary(fill_color='0.3')
m.shadedrelief()

# Draw parallels and meridians.
m.drawparallels(np.arange(-90.,99.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))

# Calculate the point density
lonx = data.Lon.values
laty = data.Lat.values
xy = np.vstack([lonx,laty])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = lonx[idx], laty[idx], z[idx]

xmap, ymap = m(x,y)
m.scatter(xmap, ymap,c=z,s=20,edgecolor='')
ax.set_title('World UFO Sightings')
plt.savefig('World_UFO_Heatmap_300dpi.png', format='png', dpi = 300)
#plt.savefig('Sightings_World_1200dpi.png', format='png', dpi = 1200)
plt.show()

# Plot of NA sightings Heatmap!
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

data['dates'] = pd.to_datetime(data['Datetimes'], errors='coerce')   
# Calculate the point density
lonx = data.Lon[(data.dates >= pd.Timestamp('1950-01-01'))].values
laty = data.Lat[(data.dates >= pd.Timestamp('1950-01-01'))].values
xy = np.vstack([lonx,laty])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = lonx[idx], laty[idx], z[idx]
xmap, ymap = m(x, y)
m.scatter(xmap,ymap,c=z,s=30,edgecolor='')
plt.title('United States UFO Sightings')
plt.tight_layout()
plt.savefig('America_UFO_Heatmap_300dpi.png', format='png', dpi = 300)
#plt.savefig('Sightings_North_America_1200dpi.png', format='png', dpi = 1200)
plt.show()

# Is there a correlation with airport, heliport, and seaplane base location?
# Airport data obtained and sliced from:
# https://www.faa.gov/airports/airport_safety/airportdata_5010/
import pandas as pd
airports = pd.read_csv('~/Desktop/Python Code/Big Data Code/FinalProject/Airport_Data.csv')
airports = airports.dropna().reset_index(drop=True)

# Have to convert latitude and longitude to a standard scale
# for map plotting.
def convert(tude):
    multiplier = 1 if tude[-1] in ['N', 'E'] else -1
    return multiplier*sum(float(x)/60**n for n, x in enumerate(tude[:-1].split('-')))

airports['Lon'] = map(lambda l: convert(l), airports['ARPLongitude'])
airports['Lat'] = map(lambda l: convert(l), airports['ARPLatitude'])
airports['ActivationDate'] = pd.to_datetime(airports['ActivationDate'].astype('str'), errors = 'coerce')

# Plotting airports for the USA
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import geopy

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
lonx = airports.Lon[(airports.ActivationDate >= pd.Timestamp('1950-01-01'))].values
laty = airports.Lat[(airports.ActivationDate >= pd.Timestamp('1950-01-01'))].values
xy = np.vstack([lonx,laty])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = lonx[idx], laty[idx], z[idx]
xmap, ymap = m(x, y)

m.scatter(xmap,ymap,c=z,s=30,edgecolor='')
plt.title('United States Airports, Heliports, and Seaplane Bases')
plt.tight_layout()
plt.savefig('Airports.png', format='png', dpi = 300)
plt.show()

# Military Bases
# Is there any visible correlation between Sightings and Military Base location?
# Military bases obtained from (a long url...): 
# url = 'https://www.google.com/maps/d/viewer?mid=1XFBnIuaJ-71hcaDJvdmBmeXNhYM&hl=en&ll=47.323248104393414%2C-114.05083400000001&z=3'
import pandas as pd
military = pd.read_csv('~/Desktop/Python Code/Big Data Code/FinalProject/Military_Data.csv')
military = military.dropna().reset_index(drop=True)

# Plotting Military Bases in the USA
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import geopy

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
lonx = military.longitude.values
laty = military.latitude.values
xy = np.vstack([lonx,laty])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = lonx[idx], laty[idx], z[idx]
xmap, ymap = m(x, y)

m.scatter(xmap,ymap,c=z,s=60,edgecolor='black')
plt.title('United States Military Bases')
plt.tight_layout()
plt.savefig('Military_Bases.png', format='png', dpi = 300)
plt.savefig('Sightings_North_America_300dpi.png', format='png', dpi = 300)
plt.show()


# Plotting by shape for the USA
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

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

shapes = data.Shape.value_counts().index.tolist()
lonx,laty = [],[]
for shape in shapes:
    lonx.append(np.nanmean(data.Lon[data.Shape == '%s' % shape].values))
    laty.append(np.nanmean(data.Lat[data.Shape == '%s' % shape].values))

x,y=m(lonx,laty)
m.scatter(x, y, edgecolor='',s=20)
plt.title('United States UFO Sightings')
plt.tight_layout()
#plt.savefig('America_UFO_Heatmap_300dpi.png', format='png', dpi = 300)
#plt.savefig('Sightings_North_America_1200dpi.png', format='png', dpi = 1200)
plt.show()


cat_cols = ['Shape', 'Timezone', 'Weekday', 'Month', 'Color']           
for col in cat_cols:
     data[col] = data[col].astype('category')
data = data[(data['Year']>=1900) & (data['Year']<=2015)]
data = data.reset_index(drop = True)
data[['Date','Time']] = data['Datetimes'].apply(lambda x: pd.Series(x.split(' ')))
data['Coordinates'] = data[['Lon','Lat']].apply(tuple, axis=1)

# Group same shape and date.
data['Shape_Date'] = data[['Shape', 'Date']].apply(lambda x: ' '.join(x), axis=1)

# Group same shape, date, and timezone.
data['Shape_Date_Timezone'] = data[['Shape', 'Date', 'Timezone']].apply(lambda x: ' '.join(x), axis=1)

# Group same shape, date, and time.
data['Shape_Date_Time'] = data[['Shape', 'Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)

# Mass sightings grouped by Shape, Date, and Timezone
mass_local = data.groupby(['Shape_Date_Timezone']).size()
dl = {mass_local.index[i]:mass_local.values[i] for i in xrange(73344)}
data['Witnesses_ML'] = pd.Series(data['Shape_Date_Timezone']).astype('str')
data['Witnesses_ML'].replace(dl, inplace=True)

# Mass sightings grouped by Shape, Date, and Time
mass_local = data.groupby(['Shape_Date_Time']).size()
dl = {mass_local.index[i]:mass_local.values[i] for i in xrange(79655)}
data['Witnesses_MLT'] = pd.Series(data['Shape_Date_Time']).astype('str')
data['Witnesses_MLT'].replace(dl, inplace=True)

# Mass sightings grouped by only Shape and Date. Extraterrestrials would
# have FTL capabilities so this should be examined!
mass_nonlocal = data.groupby(['Shape_Date']).size()
dnl = {mass_nonlocal.index[i]:mass_nonlocal.values[i] for i in xrange(49936)}
data['Witnesses_MNL'] = pd.Series(data['Shape_Date']).astype('str')
data['Witnesses_MNL'].replace(dnl, inplace=True)


header = data.columns.tolist()
data.to_csv('UFO_Modeling_Full.csv', sep = ',', columns = header)
