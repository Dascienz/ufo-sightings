#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 13:36:40 2016

@author: Dascienz
"""

"""
All processing, parsing, cleaning and formatting of scraped data.
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing scraped data from nuforc.org. (Gathered by UFO_Spider.py)
scraped_data = "~/Desktop/Python Code/Big Data Code/FinalProject/uforeports.csv"
uforeports = pd.read_csv(scraped_data)
reports = uforeports.description

reports = reports.replace(np.nan, "NaN", regex = True)
replace = {" :":":", ": ":":",
", ":",", " ,":",", ",,":",", "National UFO Reporting Center,":""}
replace_obj = re.compile('|'.join(replace.keys()))
reports = [replace_obj.sub(lambda m: replace[m.group(0)], reports[i]) for i in xrange(len(reports))]

#New set so I don't accidentally mess up the original one.
text = reports

# Extracting Event Time Data
time_search1 = [re.search('Occurred:(.*?),Reported',text[i]) for i in xrange(len(text))]
time_search2 = [re.search('Reported:(.*?),Posted',text[i]) for i in xrange(len(text))]
time = []
for i in xrange(len(time_search1)):
    try:
        if time_search1[i].group(1) in [' ','']:
            time.append(time_search2[i].group(1))
        else:
            time.append(time_search1[i].group(1))
    except AttributeError:
        time.append("NaN")

# Extracting Location Data
place_search = [re.search('Location:(.*?),Shape:',text[i]) for i in xrange(len(text))]
place = []
for i in xrange(len(place_search)):
    try:
        place.append(place_search[i].group(1))
    except AttributeError:
        place.append("NaN")

# Extracting Shape Data
shape_search = [re.search('Shape:(.*?),Duration:',text[i]) for i in xrange(len(text))]
shape = []
for i in xrange(len(shape_search)):
    try:
        shape.append(shape_search[i].group(1))
    except AttributeError:
        shape.append("NaN")
   
# Extracting Duration Data
duration_search = [re.search('Duration:(.*?),',text[i]) for i in xrange(len(text))]
duration = []
for i in xrange(len(duration_search)):
    try:
        duration.append(duration_search[i].group(1))
    except AttributeError:
        duration.append("NaN")

# Extracting Summary Data
summary_search = [re.search("Duration:(.*).",text[i]) for i in xrange(len(text))]
summary = []
for i in xrange(len(summary_search)):
    try:
        summary.append(summary_search[i].group(1))
    except AttributeError:
        summary.append("NaN")
    
summary = [summary[i].split(",") for i in range(len(summary))]
for i in xrange(len(summary)):
    summary[i].pop(0)  
summary = [" ".join(summary[i]) for i in range(len(summary))]

# DataFrame of all extracted event, location, shape, duration, and description data.
data = pd.DataFrame(({'time': time,'place': place,'shape': shape,
                      'duration': duration, 'summary': summary}))
data['shape'] = map(lambda x: str(x).lower(), data['shape'])

# Incredibly inefficient and stupid to do this manually, but these are a few
# exceptions I could see from shape count alone.
data['shape'].replace(['changed','flare'],['changing','fireball'], inplace = True)
data.ix[data.index[39347], 'shape'] = 'oval' 
data.ix[data.index[39348], 'shape'] = 'fireball'  
data.ix[data.index[103986], 'shape'] = 'triangle' 
data['shape'] = data['shape'].replace(['other','unknown','NaN'],'unspecified') 
data['summary'] = data['summary'].replace(['',' '],'NaN')
data = data[data.summary.str.contains("NaN") == False].reset_index(drop = True)  

# Dropping extreme minority sightings for simplification of shape list
for shape in ['delta','crescent','dome','hexagon','pyramid']:
    data = data[data['shape'].str.contains("%s" % shape) == False].reset_index(drop = True) 

# CHECKPOINT 1
header = ['duration','place','shape','summary','time']
data.to_csv('UFO_Data1.csv', sep=',', columns = header)

# Place Distribution
# Code used to extract geographic coordinates of sightings.
"""
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from time import sleep

geolocator = Nominatim()
places = data['place'] 
lat, lon = [],[]
for k in range(0,len(places)):
    location = geolocator.geocode("%s" % places[k], timeout = 10) 
    sleep(1.25) # Be courteous with this parameter or they'll kick you!
    try:
        lat.append(location.latitude)
        lon.append(location.longitude)
    except (AttributeError, GeocoderTimedOut):
        lat.append(np.nan)
        lon.append(np.nan)

coords = pd.DataFrame(({'lat': lat, 'lon': lon}))
header = ['lat','lon']
coords.to_csv('Coordinates_full.csv', sep = ',', columns = header)
"""

# Event Time Data
# This step must be performed to use the errors = 'coerce' step from pandas.
import re
t = data['time']
t_search = [re.search("Entered as: (.*)\)", t[i]) for i in range(len(t))]
t_new = []
for i in xrange(len(t_search)):
    try:
        t_new.append(t_search[i].group(1))
    except:
        t_new.append("%s" % t[i])

data = pd.DataFrame({'duration': data['duration'],
                     'time': t_new,
                     'place': data.place,
                     'shape': data['shape'], 
                     'summary': data['summary']})
        
data['time'] = pd.to_datetime(data['time'], errors = 'coerce')

# CHECKPOINT 2
header = ['duration','time','place','shape','summary']
data.to_csv('UFO_Data2.csv', sep = ',', columns = header)

# Still some more processing that needs to be done to clean up the data...
data = pd.read_csv('~/Desktop/BigData/FinalProject/UFO_Data2.csv')
data = data.reset_index(drop = True)

# I need to figure out which shape categories the unspecified-shape sightings
# belong in. This is really dependent on how detailed the eye witnesses
# were in their descriptions. I'm sticking with the shape categories determined
# by the NUFORC.
shapes = data['shape'].value_counts().index.tolist()
shapes.pop(1) # get rid of unspecified since we don't want to match that category.
text = data[data['shape'] == 'unspecified']['summary']
text_target = text.index.tolist()

# Awesome tools for analyzing term frequency and inverse document frequency.
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

lem = WordNetLemmatizer()

def my_tokenizer(words):
    return [lem.lemmatize(word,'n').lower() for
            word in nltk.word_tokenize(words) if len(word) > 1]

# Form a list whose items are lists of words from the unspecified sighting
# summaries. Hopefully from these lists we can pick the most likely shape
# according to our list of known labels.
words = []
for i in xrange(len(text)):
    try:
        words.append(my_tokenizer(text[text_target[i]]))
    except UnicodeDecodeError:
        words.append(my_tokenizer(text[text_target[i]].decode('latin-1')))

# There are probably better, more concise, ways of doing this. 
labels, index = [],[]
for i in xrange(len(words)):
    for j in xrange(len(words[i])):
        for shape in shapes:
            if words[i][j] == shape:
                try:
                    labels.append(shape)
                    index.append(i)
                except:
                    continue
 
text_target_index = [i for i in xrange(len(text_target))]
index_to_index = {text_target_index[i]:text_target[i] for i in xrange(len(text_target))}

labeled_df = pd.DataFrame({'index': index, 'label': labels})
labeled_df = labeled_df.replace({'index': index_to_index})
test = labeled_df.groupby('index').head(1)
test = test.set_index(test['index'])

for i, j in zip(test['index'], test['label']):
    data.ix[i, 'shape'] = j  
data['shape'] = data['shape'].replace('unspecified','other') 

# Let's extract color data using similar routines.
# List of common colors.
color_list = ['red','green','blue','orange','yellow','pink',
              'white','gray','grey','black','brown','silver',
              'gold','bronze','purple','violet']

text = data['summary']
text_target = text.index.tolist()

words = []
for i in xrange(len(text)):
    try:
        words.append(my_tokenizer(text[text_target[i]]))
    except UnicodeDecodeError: 
        words.append(my_tokenizer(text[text_target[i]].decode('latin-1')))

labels, index = [],[]
for i in xrange(len(words)):
    for j in xrange(len(words[i])):
        for color in color_list:
            if words[i][j] == color:
                try:
                    labels.append(color)
                    index.append(i)
                except:
                    continue

text_target_index = [i for i in xrange(len(text_target))]
index_to_index = {text_target_index[i]:text_target[i] for i in xrange(len(text_target))}

labeled_df = pd.DataFrame({'index': index, 'label': labels})
test = labeled_df.groupby('index').head(1) # Most popular category match
test = test.set_index(test['index'])

for i, j in zip(test['index'], test['label']):
    data.ix[i, 'color'] = j
data['color'] = data['color'].replace('grey','gray') 

# CHECKPOINT 3
header = ['duration','time','place','shape','summary','color']
data.to_csv('UFO_Data3.csv', sep = ',', columns = header)

# Still need to somehow deal with formatting this time duration data.
# I apologize that this part is so messy.
import re

# Split durations up by spaces.
durations = []
for item in data['duration']:
    try:
        durations.append(item.split())
    except AttributeError:
        durations.append(np.nan)

# Function used to filter items in list
def pickFilter(list, filter):
    return [l for l in list for m in (filter(l),) if m]

# Function used to search for numbers 0-9 and first letters m,h,s.
# I want a list containing N*(m,s,h) entries for each sighting 
# where m = 1., s = 1./60., and h = 60.
# N would be the ballpark number provided by the witness provided.

# Original Entry: approx. 5-10 sec.
# Transformed Entry: 5*s -> 5./60.
def durationFormat(s):
    searchRegex0 = re.compile('([0-9]$)').search
    searchRegex1 = re.compile('([m,h,s])').search
    x0 = pickFilter(s, searchRegex0)
    x1 = pickFilter(s, searchRegex1)
    return [x0, x1]

t = []
for item in durations:
    try:
        t.append(durationFormat(item))
    except TypeError:
        t.append(item)
# Get rid of annoying empty slots.
for i in xrange(len(t)):
        try:
            t[i][0].pop(1)
            t[i][1].pop(1)
        except (IndexError, TypeError):
            pass
t0 = []
for i in xrange(len(t)):
        try:
            t0.append(t[i][0][0])
        except (IndexError, TypeError):
            t0.append(np.nan)
t0_new = []
for item in t0:
    try:
        t0_new.append(item.split('-'))
    except AttributeError:
        t0_new.append(np.nan)
t0 = []
for i in xrange(len(t0_new)):
    try:
        t0.append(t0_new[i][0])
    except TypeError:
        t0.append(np.nan)
t1 = []
for i in xrange(len(t)):
        try:
            t1.append(t[i][1][0])
        except (IndexError, TypeError):
            t1.append(np.nan)
t2 = [] # form a list of letters s, m, and h to convert these into time units.
for i in range(len(t1)):
    try:
        t2.append(t1[i][0])
    except TypeError:
        t2.append(np.nan)
t2 = map(lambda x: str(x).lower(), t2)
# Keep time units in minutes.
t_values = {'h': 60.0, 'm': 1.0, 's': 1.0/60.0, '6': 60.0, '1': 10.0/60.0,
            'c': np.nan}

# Transformed the duration column into two columns.
duration_df = pd.DataFrame({'t0': t0, 't1': t2})
duration_df = duration_df.replace({'t1': t_values, 't0': {'~':''}})

# Now I can convert everything to numbers.
num0 = []
for n in duration_df['t0']:
    try:
        num0.append(float(n))
    except (TypeError, ValueError):
        num0.append(np.nan)

num1 = []
for n in duration_df['t1']:
    try:
        num1.append(float(n))
    except:
        num1.append(np.nan)
# This is what I needed...
duration = []
for i, j in zip(num0, num1):
    try:
        duration.append(i*j)
    except:
        duration.append(np.nan)

# It was very ugly, but it seemed to do an okay job.
data = pd.DataFrame({'Duration': duration,
                    'Date': data['time'],
                    'Place': data['place'],
                    'Shape': data['shape'],
                    'Summary': data['summary'],
                    'Color': data['color']})       
    
# CHECKPOINT 4
header = ['Duration','Date','Place','Shape','Summary','Color']
data.to_csv('UFO_Data4.csv', sep = ',', columns = header)

# FINAL POLISHING
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

data = pd.read_csv(wdir + 'UFO_Data4.csv')
coordinates = pd.read_csv(wdir + 'Coordinates_full.csv')
data = data.reset_index(drop = True)

# Columns are formatted, but can still be made more approachable for analysis.
reports2 = pd.DataFrame({'Duration': data['Duration'],
                        'Date': data['Date'].astype(str), 
                        'Lat': coordinates['lat'],
                        'Lon':coordinates['lon'], 
                        'Shape': data['Shape'],
                        'Color': data['Color'],
                        'Summary': data['Summary']})

# Want to split up date column into day, month, and year data.
def date_time(string_date):
    return datetime.datetime.strptime(string_date, "%Y-%m-%d %H:%M:%S")

time_tuples = []
for times in reports2.Date:
    try:
        date_time(times)
        time_tuples.append(date_time(times).timetuple())
    except (ValueError, TypeError):
        time_tuples.append(np.nan)


Year, Month, Day, Hour = [],[],[],[]
for i in xrange(len(time_tuples)):
    try:
        Year.append(time_tuples[i][0])
        Month.append(time_tuples[i][1])
        Day.append(time_tuples[i][2])
        Hour.append(time_tuples[i][3])
    except (ValueError, TypeError):
        Year.append(np.nan)
        Month.append(np.nan)
        Day.append(np.nan)
        Hour.append(np.nan)

Weekday = []
for i in xrange(len(time_tuples)):
    try:
        result = datetime.date(Year[i], Month[i], Day[i])
        Weekday.append(result.strftime("%A"))
    except (ValueError, TypeError):
        Weekday.append(np.nan)    

# Redefine reports DataFrame with Dates separated into Year, Month, and Day.
reports2 = pd.DataFrame({'Duration': data['Duration'],
                        'Year': Year,
                        'Month': Month,
                        'Day': Day,
                        'Weekday': Weekday,
                        'Hour': Hour,
                        'Lat': coordinates['lat'],
                        'Lon':coordinates['lon'], 
                        'Shape': data['Shape'],
                        'Color': data['Color'],
                        'Summary': data['Summary']})    

header = ['Duration','Year','Month','Day','Weekday',
          'Hour','Lat','Lon','Shape','Summary','Color']
reports2.to_csv('UFO_Dataset.csv', sep = ',', columns = header)

