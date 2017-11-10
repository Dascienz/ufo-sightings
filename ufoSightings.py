"""
Created on Wed Nov  2 10:59:28 2016

@author: Dascienz
"""

"""
Mostly plotting the scraped data to understand major trends
and commonalities in the sightings.
"""
import os
cwd = os.getcwd()

import nltk
import itertools
import wordcloud
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from scipy.stats import ttest_ind
from nltk.stem.wordnet import WordNetLemmatizer

def readData():
    data = pd.read_csv(cwd + "/ufoData.csv", index_col=0)
    return data 

data = readData()                         

# SHAPE DATA
def shapePlot():
    """Sightings distribution by UFO shape."""
    shape_counts = data['shape'].value_counts()
    shape_counts.plot(kind = "barh", colormap = "ocean")
    font = {'fontsize':'larger'}
    percent_font = {'fontsize':'smaller'}
    plt.title('Sightings by Shape', **font)
    plt.xlabel('Sightings')
    plt.xlim([0,36000])
    
    #Percentages by UFO Shape
    shape_percent = [s*(100.0/shape_counts.sum()) for s in shape_counts]
    X = shape_counts.values
    Y = [i for i in range(len(shape_percent))]
    for i in range(len(shape_percent)):
        x = X[i]+500
        y = Y[i]-0.2
        plt.text(x,y,'%.2f' % shape_percent[i] + '%', **percent_font)
    plt.tight_layout()
    plt.savefig('Sightings_by_Shape_300dpi.png', format = 'png', dpi = 300)
    plt.show()

# DURATIONS DATA
def durationPlots():
    """Durations distributions by UFO shape."""
    shapes = data['shape'].value_counts().index.tolist()
    means = ['%.1f' % data['duration'][data['shape'] == shape].mean() for shape in shapes]
    means = [float(x) for x in means]
    tuples = [*zip(shapes,means)]
    tuples.sort(key = lambda x: x[1], reverse=True)
    shapes, means = zip(*tuples)
    plt.figure()
    plt.title('Average Duration by Shape')
    plt.xlabel('Duration (minutes)')
    idy = np.arange(len(shapes))
    plt.barh(idy,means,alpha=0.5,color='red')
    plt.yticks(idy,shapes)
    plt.savefig('Sightings_by_Duration_300dpi.png', format = 'png', dpi = 300)
    plt.show()

    data_to_plot = []
    for shape in shapes:
        data_to_plot.append(data['duration'][data['shape']==shape].values)
        for array in data_to_plot:
            for i in range(len(array)):
                if np.isnan(array[i]) == True:
                    array[i] = np.nanmean(array)
                    
    plt.figure()
    plt.title('Durations Box Plots')
    plt.boxplot(data_to_plot, showfliers=False)
    plt.ylim([0,125])
    idx = np.arange(1,len(shapes)+1)
    plt.xticks(idx,shapes,rotation=90)
    plt.ylabel('Duration (minutes)')
    plt.tight_layout()
    plt.savefig('Sightings_by_Duration_Boxplot_300dpi.png', format = 'png', dpi = 300)
    plt.show()
    
def tTest(group_1, group_2):
    x = data['duration'][data['shape']==group_1]
    y = data['duration'][data['shape']==group_2]
    return ttest_ind(x.dropna().values, y.dropna().values)

# DISTRIBUTION BY HOUR (WITH COLOR MAPPING SCALED BY HEIGHT)
def hourPlot():
    """Sightings distribution by hour of the day."""
    total = data['hour'][(data['hour'] >= 0) & (data['hour'] <= 24)]

    cmap = plt.cm.get_cmap('autumn')
    Y,X = np.histogram(total,24,normed=1.0)
    y_span = Y.max()-Y.min()
    colors = [cmap(((y-Y.min())/y_span)) for y in Y]
    y_max = total.value_counts().sum()

    font = {'fontsize':'larger'}
    b = np.arange(0,24,1)
    plt.bar(b,Y*y_max,width=b[1]-b[0],color=colors,align='center')
    plt.xticks(b)
    plt.xlim([-0.5,23.5])
    plt.xlabel('24-hour Time')
    plt.ylabel('Sightings')
    plt.title('Sightings by Hour', **font)
    plt.tight_layout()
    plt.savefig('Sightings_by_Hour_300dpi.png', format='png',dpi=300)
    plt.show()

# DISTRIBUTION BY WEEKDAY (WITH COLOR MAPPING SCALED BY HEIGHT)
def weekdayPlot():
    """Sightings distribution by weekday."""
    total = pd.DataFrame({'weekday': data['weekday']})
    weekdays = {'Monday':1,'Tuesday':2,'Wednesday':3,
                'Thursday':4,'Friday':5,'Saturday':6,
                'Sunday':7}
    total['weekday'].replace(weekdays, inplace=True)
    total = total['weekday'][(total['weekday'] >= 1) & (total['weekday'] <= 7)]

    cmap = plt.cm.get_cmap('spring')
    Y,X = np.histogram(total,7,normed=1.0)
    y_span = Y.max()-Y.min()
    colors = [cmap(((y-Y.min())/y_span)) for y in Y]
    y_max = total.value_counts().sum()

    font = {'fontsize':'larger'}
    plt.bar(sorted(list(weekdays.values())),Y*y_max,width=1.0,
            color=colors,align='center',label=list(weekdays.keys()))
    plt.xticks(list(weekdays.values()),list(weekdays.keys()),ha='right',rotation='45')
    plt.xlim([-.5,7.5])
    plt.xlabel('Weekday')
    plt.ylabel('Sightings')
    plt.title('Sightings by Week', **font)
    plt.axis('tight')
    plt.tight_layout()
    plt.savefig('Sightings_by_Day_300dpi.png', format='png',dpi=300)
    plt.show()

# DISTRIBUTION BY MONTH
def monthPlot():
    """Sightings distribution by month."""
    total = data['month'][(data['month'] >= 1) & (data['month'] <= 12)]

    months = {'Jan':1,'Feb':2,'Mar':3,
              'Apr':4,'May':5,'Jun':6,
              'Jul':7,'Aug':8,'Sep':9,
              'Oct':10,'Nov':11,'Dec':12}

    cmap = plt.cm.get_cmap('summer')
    Y,X = np.histogram(total,12,normed=1.0)
    y_span = Y.max()-Y.min()
    colors = [cmap(((y-Y.min())/y_span)) for y in Y]
    y_max = total.value_counts().sum()
          
    font = {'fontsize':'larger'}
    plt.bar(list(months.values()),Y*y_max,width=1.0,
            color=colors,align='center',label=list(months.keys()))
    plt.xticks(list(months.values()),list(months.keys()),ha='right',rotation='45')
    plt.xlim([0.5,12.5])
    plt.xlabel('Month')
    plt.ylabel('Sightings')
    plt.title('Sightings by Month', **font)
    plt.tight_layout()
    plt.savefig('Sightings_by_Month_300dpi.png', format='png',dpi=300)
    plt.show()

# DISTRIBUTION BY YEAR
def yearPlot():
    """Sightings distribution by year."""
    total = data['year'][(data['year'] >= 1965) & (data['year'] <= 2016)]

    cmap = plt.cm.get_cmap('winter')
    Y,X = np.histogram(total,len(total.value_counts()),normed=1.0)
    y_span = Y.max()-Y.min()
    colors = [cmap(((y-Y.min())/y_span)) for y in Y]
    y_max = total.value_counts().sum()

    font = {'fontsize':'larger'}
    plt.bar(X[:-1],Y*y_max,width=X[1]-X[0],
            color=colors,align='center')
    plt.xticks(np.arange(X.min(),X.max()+1,5),ha='right',rotation='45')
    plt.xlim([1964.5,2016.5])
    plt.axvline(x=1997,color='red',linestyle='dashed')
    plt.text(1998,7000,'Google', rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Sightings')
    plt.title('Sightings by Year', **font)
    plt.axis('tight')
    plt.tight_layout()
    plt.savefig('Sightings_by_Year_300dpi.png', format='png',dpi=300)
    plt.show()

# DISTRIBUTION BY DAY OF THE YEAR
def yeardayPlot():
    """Sightings distribution by day of the year."""
    font = {'fontsize':'larger'}
    data['yearday'].hist(bins=np.arange(0,365,3), align='left',alpha=0.75)
    plt.ylabel('Sightings')
    plt.xlabel('Day of Year')
    plt.xlim([0,365])
    plt.text(4,2000,'New Years',rotation=45,color='green')
    plt.text(188,2000,'July 4th',rotation=45,color='green')
    plt.xticks(rotation='45')
    plt.title('Sightings by Day of Year', **font)
    plt.axis('tight')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('Sightings_by_DayOfYear.png', format = 'png', dpi = 300)
    plt.show()

def my_tokenizer(words):
    lem = WordNetLemmatizer()
    stopset = stopwords.words('english')
    stopset.extend((u'wa',u"n't",u'...','nuforc','ha'))
    return [lem.lemmatize(word,'n') for
            word in nltk.word_tokenize(words) if word.lower() not in stopset and len(word.lower()) >= 3]

def termCloud():
    """Generate wordcloud from summary text."""
    text = data['summary']
    wordlists = [my_tokenizer(text[i]) for i in range(len(text))]
    wordlist = pd.Series(list(itertools.chain.from_iterable(wordlists)))
    words = wordlist.value_counts().index.tolist()
    frequencies = wordlist.value_counts().values.tolist()
    word_dict = dict([*zip(words, frequencies)])

    plt.figure(figsize=(14,7))
    icon = Image.open(WDIR + "/Plots/alien_mask.png")
    mask = Image.new("RGB", icon.size, (255,255,255))
    mask.paste(icon,icon)
    alien_mask = np.array(mask)
    cloud = wordcloud.WordCloud(background_color='white',
                                mask=alien_mask)
    cloud.generate_from_frequencies(word_dict)
    plt.imshow(cloud.recolor(colormap=plt.cm.viridis))
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('UFO_Summary_Word_Cloud.png',format='png',dpi=600)

def colorCloud():
    """Generate wordcloud from color categories."""
    plt.figure(figsize=(8,6))
    colors = data['color'].value_counts().index.tolist()        
    frequencies = data['color'].value_counts().values.tolist() 
    color_dict = dict([*zip(colors, frequencies)][1:])
    cloud = wordcloud.WordCloud(background_color='white',
                                random_state=58,
                                width=1600,height=800)
    cloud.generate_from_frequencies(color_dict)
    plt.axis('off')
    plt.imshow(cloud.recolor(colormap=plt.cm.hsv))
    plt.savefig('Colors_Word_Cloud.png',format='png',dpi=300)
    plt.show()
