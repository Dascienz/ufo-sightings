# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 18:30:18 2016

@author: Dascienz
"""

import os
cwd = os.getcwd()

import datetime
import numpy as np
import pandas as pd

def readData():
    data = pd.read_csv(cwd + "/ufoData.csv", index_col=0)
    data = data.dropna()
    data = data.reset_index(drop=True)
    data = data[['shape','lat','lon','datetime','summary']]
    data['datetime'] = data['datetime'].apply(lambda x: pd.Timestamp(x).date())
    data = data[(data['datetime'] <= datetime.date(2017, 1, 1))]
    
    cols = ['shape', 'lat', 'lon', 'datetime']
    data['reports'] = data.groupby(cols).transform('count')
    data['credibility'] = np.where(data['reports']>=2, 'High', 'Low')
    return data

data = readData()

"""
Now, let's vectorize summaries and use a Support Vector Classifier (SVM) to create a binary
classification model for predicting whether a UFO sighting has a high or low credibility. 
High credibility labels were assigned to sightings with multiple reports based on 
shared shape, location, and date information. In other words, reports with multiple sightings 
have high credibility and sightings with single reports have low credibility.
"""

import sklearn
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer

#Define features (summaries) and labels (credibilities)
summaries = data['summary']
credibilities = data['credibility']

#Vectorize features, aka text summaries.
vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=5)
vectors = vectorizer.fit_transform(summaries)

f_train, f_test, l_train, l_test = train_test_split(vectors, credibilities, test_size = 0.3)
svm = LinearSVC(C=0.7, class_weight='balanced')
clf = CalibratedClassifierCV(svm)
clf.fit(f_train, l_train)

"""Training Data Predictions."""
pred_train = clf.predict(f_train)
print(classification_report(l_train, pred_train))

"""Testing Data Predictions."""
pred_test = clf.predict(f_test)
print(classification_report(l_test, pred_test))

"""Using the model: Percent chance credibilities."""
fake = ["I saw a white flying saucer heading southeast while I was out with my dog.\
         It hovered for a bit before disappearing. I haven't told anyone else about it."]

real = ["My wife and I were walking our dogs on the south side of South Mountain\
        in the mountain park when we saw the ufo come over the mountain at low\
        altitude (500'appx) headed southbound.  It was s i l e n t and had 5 white\
        globes along the leading edge of a black triangle shaped object which blacked\
        out the stars and was about a mile wide."]

def credibility(summary):
    """String needs to be in a list, e.g. [Description]"""
    s = vectorizer.transform(summary)
    y_proba = clf.predict_proba(s)
    prob = y_proba[0][0]*100
    if prob > 50.0:
        return print("High credibility: " + "%.2f" % (y_proba[0][0]*100) + "%")
    else:
        return print("Low credibility: " + "%.2f" % (y_proba[0][0]*100) + "%")
    
credibility(fake)
credibility(real)