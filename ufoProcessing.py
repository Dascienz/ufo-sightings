"""
Created on Sat Nov 12 13:36:40 2016

@author: Dascienz
"""

"""
All processing, parsing, cleaning and formatting of scraped data.
"""
import os
cwd = os.getcwd()

import re
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
First, we import the raw data scraped by ufoSpider.py. This needs to be 
cleaned and organized into a nice dataframe format.
"""
def readScraped():
    data = pd.read_csv(cwd + "/nuforcReports.csv")
    data['description'] = data['description'].astype('str')
    return data
data = readScraped()

def replace(series):
    replace = {" :":":", ": ":":", ", ":",",
               " ,":",", ",,":",", "National UFO Reporting Center,":""}
    replace_obj = re.compile('|'.join(replace.keys()))
    new_series = [replace_obj.sub(lambda x: replace[x.group(0)], series[i]) for i in range(len(series))]
    return new_series
data['description'] = replace(data['description'])

def parseReport(string, series):
    search = [re.search(str(string), series[i]) for i in range(len(series))]
    new_series = []
    for i in range(len(search)):
        try:
            new_series.append(search[i].group(1).strip())
        except AttributeError:
            new_series.append(np.nan)
    return new_series

"""
Parsing through and organizing the reports entered by eye-witnesses.
"""

t = parseReport("Occurred:(.*?),Reported", data['description'])
find = "Entered as:(.*)\)"
search = [re.search(find, str(t[i])) for i in range(len(t))]
time = []
for i in range(len(search)):
    try:
        time.append(search[i].group(1))
    except:
        time.append(str(t[i]))

place = parseReport("Location:(.*?),Shape:", data['description'])

shape = parseReport("Shape:(.*?),Duration:",data['description'])
shape = [str(s).lower() for s in shape]

duration = parseReport("Duration:(.*?),",data['description'])
summary = parseReport("^(?:[^,]*\,){8}(.*)",data['description'])


"""
DataFrame of all extracted event, location, shape, duration, and summary information.
Further clean up is also necessary.
"""
data = pd.DataFrame(({'time': time,'place': place,'shape': shape,
                      'duration': duration, 'summary': summary}))

data['shape'].replace(['changed','flare','triangular','round'],
                      ['changing','fireball','triangle', 'oval'], inplace = True)
data.loc[data.index[39348], 'shape'] = 'fireball'
data['shape'] = data['shape'].replace(['other','unknown'], 'unspecified')
data['summary'] = data['summary'].replace(['',' '], 'NaN')
data = data[data['summary'].str.contains("NaN") == False].reset_index(drop=True)
data['time'] = pd.to_datetime(data['time'], errors = 'coerce')

counts = data['shape'].value_counts()
data = data[data['shape'].isin(counts[counts > 300].index)].reset_index(drop=True)


"""
For collecting latitude and longitude coordinates from the location data.

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from time import sleep

geolocator = Nominatim()
lat, lon = [],[]
for k in range(0,len(data['place'])):
    location = geolocator.geocode(str(data['place'][k]), timeout = 50) 
    sleep(1.5) #Need to be courteous with this parameter or the server will kick you.
    try:
        lat.append(location.latitude)
        lon.append(location.longitude)
    except (GeocoderTimedOut, AttributeError):
        lat.append(location.latitude)
        lon.append(location.longitude)
    print(k)
coords = pd.DataFrame(({'lat': lat, 'lon': lon}))
header = ['lat','lon']
coords.to_csv(cwd + '/locationData.csv', sep = ',', columns = header)
"""

"""
There's a significant number of ill-defined (unknown, other) shape entries which we 
need to figure out from the information provided by the eye-witnesses. 
"""

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

def my_tokenizer(words):
    lem = WordNetLemmatizer()
    stopset = stopwords.words('english')
    stopset.extend((u'wa',u"n't",u'...','nuforc','ha'))
    return [lem.lemmatize(word,'n') for
            word in nltk.word_tokenize(words) if word.lower() not in stopset and len(word.lower()) >= 3]

def featureExtract(text, labels):
    """ 
    Extracts feature label based on frequency of appearance of a known label in text.
    Text should be a dataframe column of string data!
    """
    words = []
    idx = text.index.tolist()
    for i in range(len(text)):
        try:
            words.append(my_tokenizer(text[idx[i]].lower()))
        except (UnicodeDecodeError):
            words.append(my_tokenizer(text[idx[i]].lower().decode('latin-1')))

    x, y = [],[]
    for i in range(len(words)):
        matches = []
        for word in words[i]:
            if word in labels:
                matches.append(word)
        y.append(matches)
        x.append(idx[i])
                   
    for i in range(len(y)):
        d = sorted({k:y[i].count(k) for k in y[i]}.items(), key=lambda s: s[1])
        if y[i] != []:
            y[i] = d[-1][0]
        elif y[i] == []:
            y[i] = 'unspecified'
    return x, y

data['shape'] = data['shape'].replace(['other','unknown'],'unspecified') 
shapes = data['shape'].value_counts().index.tolist()
text = data[data['shape'] == 'unspecified']['summary'].astype('str')

index,label = featureExtract(text, shapes)
for i, j in zip(index, label):
    data.loc[i, 'shape'] = j  
data['shape'] = data['shape'].replace('unspecified','other') 

"""
We can extract color data using similar methods. This routine takes a few minutes.
"""
colors = ['red','green','blue','orange','yellow','pink','magenta',
          'white','gray','grey','black','brown','silver','indigo',
          'gold','bronze','purple','violet','rainbow','tan']

text = data['summary'].astype('str')

index, label = featureExtract(text, colors)
data['color'] = pd.Series('unspecified', index=data.index)
for i, j in zip(index, label):
    data.loc[i, 'color'] = j
data['color'] = data['color'].replace('grey','gray') 

"""
The duration data is full of typos and various formattings. Let's clean that up!
"""
def durationClean(s):
    numeric_list = ["(\d+)"]
    unit_list = ["se[cs]", "secon[ds]",
                 "mi[ns]", "mi[mn]ut[es]",
                 "h[rs]", "hou[rs]"]
    numerics = re.compile("|".join(str(x) for x in numeric_list))
    units = re.compile("|".join(str(x) for x in unit_list))
    match_1 = re.search(numerics, s).group(0)
    match_2 = re.search(units, s.lower()).group(0)[0]
    return [match_1, match_2]

durations = []
for item in data['duration']:
    try:
        durations.append(durationClean(item))
    except AttributeError:
        durations.append(np.nan)

conversion = {'s': 1.0/60.0, 'm': 1.0, 'h': 60.0, '6': 60.0}
one, two = [],[]
for x in durations:
    try:
        one.append(x[0])
        two.append(x[1])
    except:
        one.append(np.nan)
        two.append(np.nan) 
df = pd.DataFrame({'1': one,'2': two}) 
df['2'].replace(conversion, inplace=True)

duration = []
for i, j in zip(df['1'], df['2']):
    try:
        duration.append(float(i)*float(j))
    except:
        duration.append(np.nan)
      
"""
Location data obtained from geopy.
"""

def readCoordinates():
    coordinates = pd.read_csv(cwd + "/locationData.csv", index_col=0)
    return coordinates

coordinates = readCoordinates()

"""
Further organization of the date-time data.
"""
time_tuples = []
for times in data['time']:
    try:
        time_tuples.append(times.timetuple())
    except (ValueError, TypeError):
        time_tuples.append(np.nan)

Year, Month, Day, Hour, Yearday = [],[],[],[],[]
for i in range(len(time_tuples)):
    try:
        Year.append(time_tuples[i][0])
        Month.append(time_tuples[i][1])
        Day.append(time_tuples[i][2])
        Hour.append(time_tuples[i][3])
        Yearday.append(time_tuples[i][7])
    except (ValueError, TypeError):
        Year.append(np.nan)
        Month.append(np.nan)
        Day.append(np.nan)
        Hour.append(np.nan)
        Yearday.append(np.nan)

Weekday = []
for i in range(len(time_tuples)):
    try:
        result = datetime.date(Year[i], Month[i], Day[i])
        Weekday.append(result.strftime("%A"))
    except (ValueError, TypeError):
        Weekday.append(np.nan)    

# Redefine reports DataFrame with Dates separated into Year, Month, and Day.
reports = pd.DataFrame({'duration': duration,
                        'datetime': data['time'],
                        'year': Year,
                        'month': Month,
                        'day': Day,
                        'hour': Hour,
                        'weekday': Weekday,
                        'yearday': Yearday,
                        'lat': coordinates['lat'],
                        'lon': coordinates['lon'], 
                        'shape': data['shape'],
                        'color': data['color'],
                        'summary': data['summary']})    

def exportData(df):
    header = df.columns.tolist()
    df.to_csv(cwd + "/ufoData.csv", sep = ',', columns = header)

exportData(reports)
