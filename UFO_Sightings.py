# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:59:28 2016

@author: Dascienz
"""

"""
Mostly plotting the scraped data to understand major trends
and commonalities in the sightings.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cleaned_data = "~/Desktop/Python Code/Big Data Code/FinalProject/UFO_Dataset.csv"
reports = pd.read_csv(cleaned_data)
del reports['Unnamed: 0']
reports = reports.reset_index(drop = True)  

# Dictionary for changing weekday categories into number tags.
days = {'Monday':1,'Tuesday':2,'Wednesday':3,
        'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}

# Dictionary for changing color categories into number tags.
color_labels = sorted(reports.Color.value_counts().index.tolist())
colors = {color_labels[i]:i for i in xrange(len(color_labels))}
        
# Dictionary for changing shape categories into number tags.
shape_labels = sorted(reports.Shape.value_counts().index.tolist())
shapes = {shape_labels[i]:i for i in xrange(len(shape_labels))}

# SHAPE DATA
shape_counts = reports.Shape.value_counts()
shape_counts.plot(kind = "barh", colormap = "ocean")
plt.xlabel('No. Sightings')
plt.title('Sightings by Shape')
plt.tight_layout()
plt.savefig('Sightings_by_Shape_300dpi.png', format = 'png', dpi = 300)
#plt.savefig('Sightings_by_Shape_1200dpi.png', format = 'png', dpi = 1200)
plt.show()

shape_counts = reports.Shape.value_counts().tolist()
shape_labels = reports.Shape.value_counts().index.tolist()
shape_percent = [shape_counts[i]*(100.0/reports.Shape.value_counts().sum()) for i in xrange(len(shape_counts))]
for i in xrange(len(shape_percent)):
    print '%s ==>' % shape_labels[i],'%.2f' % shape_percent[i],'%'

# DURATIONS DATA
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

ts, stds = [],[]
for shape in shapes:
    mean = np.mean(reports.Duration[reports.Shape == '%s' % shape].dropna())
    ts.append(mean)
    std = np.std(reports.Duration[reports.Shape == '%s' % shape].dropna())
    stds.append(std)

lst=[]
for t,std,s in zip(ts,stds,shapes):
    lst.append((t,s))
    
lst = sorted(lst)    
v,l = zip(*lst)
x = np.arange(len(v)) 

plt.barh(x,v,align='center',color='tomato')
plt.axis('tight')
plt.xticks(np.arange(0,31,5))
plt.yticks(x,l)
plt.xlabel('Duration (minutes)')
plt.ylabel('Shape')
plt.xlim([0,30])
plt.title('Average Duration by Shape')
plt.tight_layout()
plt.savefig('Average_Duration_by_Shape_300dpi.png', format = 'png', dpi = 300)
#plt.savefig('Sightings_by_Shape_1200dpi.png', format = 'png', dpi = 1200)
plt.show()

# Plot some normal distributions for durations data. 
bin1 = ['changing','cylinder','formation',
        'triangle','diamond','oval','egg']    
bin2 = ['light', 'fireball', 'flash', 'other','cone',
        'circle', 'sphere', 'disk']
bin3 = ['teardrop','cigar','chevron','cross','rectangle']

shapes = bin1+bin2+bin3

for shape in bin1:
    print 'Showing for Shape = {}'.format(shape)
    mean = np.mean(reports.Duration[reports.Shape == '%s' % shape].dropna())
    std = np.std(reports.Duration[reports.Shape == '%s' % shape].dropna())
    range = np.arange(-300, 300, 1)
    plt.hold(True)
    plt.plot(range, norm.pdf(range, mean, std),
             linewidth = 2.0,
             label = '{}'.format(shape)+r' ($\mu$ = %.1f min)' % (mean))
    plt.title('Duration Distributions: Set One')
    plt.grid(True)
    plt.legend(loc = 'upper left', ncol= 1, prop = {'size':9.5})
plt.tight_layout()
plt.savefig('Durations_by_Shape_1.png', format = 'png', dpi = 300)
plt.show()

for shape in bin2:
    print 'Showing for Shape = {}'.format(shape)
    mean = np.mean(reports.Duration[reports.Shape == '%s' % shape].dropna())
    std = np.std(reports.Duration[reports.Shape == '%s' % shape].dropna())
    range = np.arange(-300, 300, 1)
    plt.hold(True)
    plt.plot(range, norm.pdf(range, mean, std),
             linewidth = 2.0, 
             label = '{}'.format(shape)+r' ($\mu$ = %.1f min)' % (mean))
    plt.title('Duration Distributions: Set Two')
    plt.grid(True)
    plt.legend(loc = 'upper left', ncol= 1, prop = {'size':9.5})
plt.tight_layout()
plt.savefig('Durations_by_Shape_2.png', format = 'png', dpi = 300)
plt.show()

for shape in bin3:
    print 'Showing for Shape = {}'.format(shape)
    mean = np.mean(reports.Duration[reports.Shape == '%s' % shape].dropna())
    std = np.std(reports.Duration[reports.Shape == '%s' % shape].dropna())
    range = np.arange(-300, 300, 1)
    plt.hold(True)
    plt.plot(range, norm.pdf(range, mean, std),
             linewidth = 2.0, 
             label = '{}'.format(shape)+r' ($\mu$ = %.1f min)' % (mean))
    plt.title('Duration Distributions: Set Three')
    plt.grid(True)
    plt.legend(loc = 'upper left', ncol= 1, prop = {'size':9.5})
plt.tight_layout()
plt.savefig('Durations_by_Shape_3.png', format = 'png', dpi = 300)
plt.show()


# DISTRIBUTION BY HOUR (WITH COLOR MAPPING SCALED BY HEIGHT)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

total = reports.Hour[(reports.Hour >= 0) & (reports.Hour <= 24)]

cmap = plt.cm.get_cmap('autumn')
Y,X = np.histogram(total,24,normed=1.0)
y_span = Y.max()-Y.min()
colors = [cmap(((y-Y.min())/y_span)) for y in Y]
y_max = total.value_counts().sum()

b = np.arange(0,24,1)
plt.bar(b,Y*y_max,width=b[1]-b[0],color=colors,align='center')
plt.xticks(b)
plt.xlim([-0.5,23.5])
plt.xlabel('24-hour Time')
plt.ylabel('No. Sightings')
plt.title('Sightings by Hour')
plt.tight_layout()
plt.savefig('Sightings_by_Hour_300dpi.png', format='png',dpi=300)
plt.show()

# DISTRIBUTION BY WEEKDAY (WITH COLOR MAPPING SCALED BY HEIGHT)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

total = pd.DataFrame({'Weekday':reports.Weekday})
total.Weekday.replace(days, inplace=True)
total = total.Weekday[(total.Weekday >= 1) & (total.Weekday <= 7)]

cmap = plt.cm.get_cmap('spring')
Y,X = np.histogram(total,7,normed=1.0)
y_span = Y.max()-Y.min()
colors = [cmap(((y-Y.min())/y_span)) for y in Y]
y_max = total.value_counts().sum()

plt.bar(sorted(days.values()),Y*y_max,width=1.0,
        color=colors,align='center',label=days.keys())
plt.xticks(days.values(),days.keys(),ha='right',rotation='45')
plt.xlim([-.5,7.5])
plt.xlabel('Day of the Month')
plt.ylabel('No. Sightings')
plt.title('Sightings by Day')
plt.axis('tight')
plt.tight_layout()
plt.savefig('Sightings_by_Day_300dpi.png', format='png',dpi=300)
plt.show()

# DISTRIBUTION BY MONTH (WITH COLOR MAPPING SCALED BY HEIGHT)
total = reports.Month[(reports.Month >= 1) & (reports.Month <= 12)]

months = {'Jan':1,'Feb':2,'Mar':3,
          'Apr':4,'May':5,'Jun':6,
          'Jul':7,'Aug':8,'Sep':9,
          'Oct':10,'Nov':11,'Dec':12}

cmap = plt.cm.get_cmap('summer')
Y,X = np.histogram(total,12,normed=1.0)
y_span = Y.max()-Y.min()
colors = [cmap(((y-Y.min())/y_span)) for y in Y]
y_max = total.value_counts().sum()
          
plt.bar(months.values(),Y*y_max,width=1.0,
        color=colors,align='center',label=months.keys())
plt.xticks(months.values(),months.keys(),ha='right',rotation='45')
plt.xlim([0.5,12.5])
plt.xlabel('Month')
plt.ylabel('No. Sightings')
plt.title('Sightings by Month')
plt.tight_layout()
plt.savefig('Sightings_by_Month_300dpi.png', format='png',dpi=300)
plt.show()

# DISTRIBUTION BY YEAR (WITH COLOR MAPPING SCALED BY HEIGHT)
total = reports.Year[(reports.Year >= 1965) & (reports.Year <= 2016)]

cmap = plt.cm.get_cmap('winter')
Y,X = np.histogram(total,len(total.value_counts()),normed=1.0)
y_span = Y.max()-Y.min()
colors = [cmap(((y-Y.min())/y_span)) for y in Y]
y_max = total.value_counts().sum()

plt.bar(X[:-1],Y*y_max,width=X[1]-X[0],
        color=colors,align='center')
plt.xticks(np.arange(X.min(),X.max()+1,5),ha='right',rotation='45')
plt.xlim([1964.5,2016.5])
plt.xlabel('Year')
plt.ylabel('No. Sightings')
plt.title('Sightings by Year')
plt.axis('tight')
plt.tight_layout()
plt.savefig('Sightings_by_Year_300dpi.png', format='png',dpi=300)
plt.show()

# SUMMARY DATA
# Basic tools for analyzing term frequency and inverse document frequency.
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize

# Modified lemmatizer from class!
def my_tokenizer(words):
    lem = WordNetLemmatizer()
    return [lem.lemmatize(word,'n').lower() for
            word in nltk.word_tokenize(words) 
            if len(word) >= 3]

text = reports.Summary

wordlist = [my_tokenizer(text[i].decode('latin-1')) for i in xrange(len(text))]
            
from __future__ import division
def lexical_diversity(text):
    ld = len(set(text))/len(text)
    return ld

lexical = [lexical_diversity(wordlist[i]) for i in range(len(wordlist))]

words, freqs = [], []
for item in wordlist:
    for word in item:
        freqs.append(item.count(word))
        words.append(word)
word_tuples = zip(words, freqs)

from itertools import groupby
from operator import itemgetter
label = itemgetter(0)
value = itemgetter(1)
word_tuples_ = [(k, sum(item[1] for item in tuples))
        for k, tuples in groupby(sorted(word_tuples, key=label), key=label)]

word_tuples_ = sorted(word_tuples_, key = lambda x:x[1], reverse = True)

# Filter out some more stop_words
# I added this step after first viewing the wordcloud.
word_tuples_sw = []
stop_words = stopwords.words('english') # nltk stopwords
stop_words.extend((u"wa",u"n't",u"...","nuforc","ha")) # words nobody should want in their wordcloud. ha wa!
for word in word_tuples_:
    if word[0] in stop_words:
        pass
    else:
        word_tuples_sw.append(word)
        
# Generate a word cloud from the UFO summary data.
import wordcloud
cloud = wordcloud.WordCloud(width=1600, height=800)
word_cloud = cloud.generate_from_frequencies(word_tuples_sw)
plt.figure(figsize=(20,10))
plt.imshow(word_cloud)
plt.axis('off')
plt.tight_layout(pad=0)
#plt.savefig('UFO_Summary_Word_Cloud.png',format='png',dpi=300)

# COLOR DATA!
# List of common colors.
color_list = ['red','green','blue','orange','yellow','pink','white','gray','grey',
          'black','brown','silver','gold','bronze','purple','violet']
            
colors, color_freqs = [],[]
for item in wordlist:
    for word in item:
        if word in color_list:
            colors.append(word)
            color_freqs.append(item.count(word))
        else:
            continue

# Change spelling of 'grey' to 'gray' for consistency amongst 'colours'.
colors = ['gray' if color == 'grey' else color for color in colors]
        
color_tuples = zip(colors, color_freqs)

from itertools import groupby
from operator import itemgetter
label = itemgetter(0)
value = itemgetter(1)
color_tuples_ = [(k, sum(item[1] for item in tuples))
        for k, tuples in groupby(sorted(color_tuples, key=label), key=label)]
            
# Generate a word cloud from the UFO Color data.
import wordcloud

cloud = wordcloud.WordCloud(background_color='white',
                            random_state=67,
                            width=1600, height=800)
cloud.generate_from_frequencies(color_tuples_)
plt.figure(figsize=(8,6))
plt.imshow(cloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('Colors_Word_Cloud.png',format='png',dpi=300)

            
# TOTAL WORLD REPORTS MAP
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])

m = Basemap(projection='kav7',lon_0 = 0,resolution = None)
m.drawmapboundary(fill_color='0.3')
m.shadedrelief()

# Draw parallels and meridians.
m.drawparallels(np.arange(-90.,99.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))

x, y = m(reports.Lon.values, reports.Lat.values)
m.plot(x, y, 'o', color='Red', markersize = 3.0)
ax.set_title('World UFO Sightings')
plt.savefig('Sightings_World_300dpi.png', format='png', dpi = 300)
#plt.savefig('Sightings_World_1200dpi.png', format='png', dpi = 1200)
plt.show()


# Revised Dataset for further processing. 
model = pd.DataFrame({'Duration': reports.Duration,
                        'Year': reports.Year,
                        'Month': reports.Month,
                        'Day': reports.Day,
                        'Weekday': reports.Weekday,
                        'Hour': reports.Hour,
                        'Lat': reports.Lat,
                        'Lon': reports.Lon, 
                        'Shape': reports.Shape,
                        'Color': reports.Color}) 

model.Weekday.replace(days, inplace=True)
model.Color.replace(colors, inplace=True)
model.Shape.replace(shapes, inplace=True) 

# For this DataFrame I'm dropping all rows that contain a null value.
#model_1 = model.dropna().reset_index(drop = True)

# For this DataFrame I'm filling null values with mean() based on shape
# category.
model_2 = model.groupby('Shape').transform(lambda x: x.fillna(x.mean()))
model_2['Shape'] = pd.Series(model['Shape'], index = model.index)

# Numbers for year, month, day, weekday, and color should be integers.
decimals = pd.Series([0, 0, 0, 0, 0,0],index=['Year', 'Month', 'Day', 'Hour', 'Weekday','Color'])                   
model_2 = model_2.round(decimals)

# Need to convert from local time to UTC to make accurate timestamps. 
# To do this, I need timezone information for all of the local datetimes.
# I can find timezones from the latitude and longitude coordinates.
# Didn't time it, but this takes a decent chunk of time ~ 30 min.
from tzwhere import tzwhere

where = tzwhere.tzwhere()
timezones = []
for i,j in zip(model_2.Lat, model_2.Lon):
    try:
        timezones.append(where.tzNameAt(i,j))
    except (AttributeError, TypeError):
        timezones.append(np.nan)

model_2['Timezone'] = timezones
model_2['Dates'] = dates

# Let's save this augmented dataset so we don't lose the timezones from 
# an unexpected restart.
#header = model_2.columns.tolist()
#model_2.to_csv('UFO_Dataset_Modeling.csv', sep = ',', columns = header)

#Checkpoint for importing data
import pandas as pd
import numpy as np
model = pd.read_csv('~/Desktop/BigData/FinalProject/UFO_Dataset_Modeling.csv')
del model['Unnamed: 0'] # This freakin' column! Why?!
model = model.reset_index(drop = True)
        
model['Datetimes'] = pd.to_datetime(model[['Year', 'Month', 'Day', 'Hour']])
temp = model[model.Dates.isnull()].index.values.tolist()
model = model.replace({'Dates' : {model.Dates[i]:model.Datetimes[i] for i in temp}})

import calendar
import pytz
import datetime

def date_time(string_date):
    return datetime.datetime.strptime(string_date, "%Y-%m-%d %H:%M:%S")

def time_stamp(date,timezone):
    datetime = date_time(date)
    tz = pytz.timezone('%s'%timezone)             
    dt = tz.normalize(tz.localize(datetime, is_dst = True))
    return calendar.timegm(dt.utctimetuple())

timestamps = []
for i in xrange(len(model.Datetimes)):
    try:
        timestamps.append(time_stamp(model.Datetimes[i],model.Timezone[i]))
    except:
        timestamps.append(np.nan)

model['Timestamps'] = timestamps


# DISTRIBUTION BY DAY OF THE YEAR
# This is at the end because it took me a while to figure out
# how to get the day of year from datetime objects... Simpler than I thought.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime

model['Month'] = model['Month'].astype('str')
model['Day'] = model['Day'].astype('str')
model['DayOfYear'] = model[['Month','Day']].apply(lambda x: '-'.join(x), axis=1)
model['DayOfYear'] = pd.to_datetime(model[['Year','Month', 'Day']], format = "%y-%m-%d")
model['DayOfYear'] = model['DayOfYear'].apply(lambda x: x.timetuple().tm_yday)

model.DayOfYear.hist(bins = np.arange(1,366,10))
plt.ylabel('No. Sightings')
plt.xlabel('Day of Year')
plt.xlim([.5,366.5])
plt.xticks(rotation='45')
plt.title('Sightings by Day of Year')
plt.axis('tight')
plt.tight_layout()
plt.savefig('Sightings_by_DayOfYear.png', format = 'png', dpi = 300)
plt.show()

header = model.columns.tolist()
model.to_csv('UFO_Dataset_Modeling.csv', sep = ',', columns = header)



