# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 18:30:18 2016

@author: Dascienz
"""

"""
1.) Correlation matrix of UFO features.

2.) Shape prediction based on different features using Logistic Regression and 
Decision Tree Classification methods.
"""

import pandas as pd
import numpy as np

model = pd.read_csv('~/Desktop/Python Code/Big Data Code/FinalProject/UFO_Modeling_Full.csv')
del model['Unnamed: 0']
model = model.reset_index(drop = True)

# Dictionary for changing weekday categories into number tags.
weekdays = {'Monday':1,'Tuesday':2,'Wednesday':3,
        'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
        
# Vise versa
inv_weekdays = {v: k for k, v in weekdays.iteritems()}

# Dictionary for changing shape categories into number tags.
shape_labels = sorted(model.Shape.value_counts().index.tolist())
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
color_labels = sorted(model.Color.value_counts().index.tolist())
colors = {color_labels[i]:i for i in xrange(len(color_labels))}

# Vise versa
inv_colors = {v: k for k, v in colors.iteritems()}    

# Dictionary for changing timezone categories into number tags and vise versa.
timezone_labels = sorted(model.Timezone.value_counts().index.tolist())
timezones = {timezone_labels[i]:i for i in xrange(len(timezone_labels))}

# Vise versa
inv_timezones = {v: k for k, v in timezones.iteritems()}  

model_n = model.replace({"Weekday": weekdays,
                     "Shape": shapes,
                     "Month": months,
                     "Color": colors,
                     "Timezone": timezones}) 

# Correlation Matrix Function
num_data = pd.DataFrame({'Day': model_n.Day,'Duration': model_n.Duration,
                         'Hour': model_n.Hour, 'Month': model_n.Month,
                         'Lat': model_n.Lat,'Lon': model_n.Lon, 
                         'Year': model_n.Year, 'Shape': model_n.Shape,
                         'Lexical': model_n.Lexical_Diversity,
                         'Timezone': model_n.Timezone,'Color': model_n.Color,
                         'Witnesses': model.Witnesses_ML})

num_data = (num_data - num_data.mean())/(num_data.max()-num_data.min())
                        
def correlation_matrix(x):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax.imshow(x.corr(),interpolation='nearest', cmap=cmap)
    ax.grid(True)
    plt.axis('tight')
    plt.tight_layout()
    plt.title('UFO Features Correlation Matrix')
    labels = [x.columns[i] for i in range(len(x.columns))]
    labels_dict = {labels[i]:i for i in range(len(labels))}
    plt.xticks(labels_dict.values(),labels_dict.keys(),
               rotation='45', fontsize = 10.0)
    plt.yticks(labels_dict.values(),labels_dict.keys(),
               rotation='45', fontsize = 10.0)
    # Add colorbar, specify tick locations to match desired ticklabels
    fig.colorbar(cax)
    plt.tight_layout(pad=0)
    plt.savefig('Correlation_Matrix.png',format = 'png', dpi = 300)
    plt.show()

correlation_matrix(num_data)

data = model[(model['Year']>=1900) & (model['Year']<=2015)]
data = data.reset_index(drop = True)

# Mass Sightings Plotting
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

# Calculate the point density
lonx = data.Lon[(data.Witnesses_ML>=1) & (data.Witnesses_ML<=558)].values
laty = data.Lat[(data.Witnesses_ML>=1) & (data.Witnesses_ML<=558)].values

xmap, ymap = m(lonx,laty)
data['xmap'] = xmap
data['ymap'] = ymap

m.scatter(data.xmap.values, 
          data.ymap.values,
          s=0.65*data.Witnesses_ML.values,
          c = 'hot',
          edgecolor='black')
#plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))

plt.title('Mass UFO Sightings')
plt.savefig('America_UFO_Heatmap_300dpi.png', format='png', dpi = 300)
#plt.savefig('Sightings_North_America_1200dpi.png', format='png', dpi = 1200)
plt.show()

# Now we need to isolate the features we're interested in.
# Is location a good feature for predicting shape categories?
# Is time a good feature for predicting location?
from sklearn.cross_validation import train_test_split
#from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np

model = pd.read_csv('~/Desktop/BigData/FinalProject/UFO_Modeling_Full.csv')
del model['Unnamed: 0']
model = model.reset_index(drop = True)

data = model[(model['Year']>=1900) & (model['Year']<=2015)]
data = data.reset_index(drop = True)
data['Shape'] = data['Shape'].replace('unspecified','other')
data = data.reindex(np.random.permutation(data.index)).reset_index(drop = True)

timezone_onehot = pd.get_dummies(data['Timezone'])
weekday_onehot = pd.get_dummies(data['Weekday'])
month_onehot = pd.get_dummies(data['Month'])
shape_onehot = pd.get_dummies(data['Shape'])
color_onehot = pd.get_dummies(data['Color'])

def norm(column):
    return np.abs((column-column.mean())/(column.max()-column.min()))

Latitude = norm(data.Lat)
Longitude = norm(data.Lon)
Hour = norm(data.Hour)
Year = norm(data.Year)
Duration = norm(data.Duration)
Witnesses = norm(data.Witnesses_ML)

features = pd.concat((Latitude, 
                      Longitude,
                      timezone_onehot,
                      weekday_onehot,
                      color_onehot,
                      month_onehot,
                      Hour,
                      Year),axis=1).as_matrix()

labels = data['Shape']

f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size = 0.20)

lr = LogisticRegression(class_weight='balanced')
lr.fit(f_train, l_train)

# Make LogisticRegression Predictions
expected = l_test
predicted = lr.predict(f_test)
print "\n Precision score: ", precision_score(expected, predicted, average = 'weighted')*100.00, "%"

# summarize the fit of the model to file
import datetime
x = [datetime.datetime.now().year,
     datetime.datetime.now().month,
     datetime.datetime.now().day,
     datetime.datetime.now().hour,
     datetime.datetime.now().minute,
     datetime.datetime.now().second]

f = open('LR_Report%s.txt' % (''.join(str(s) for s in x)),'w')
print >>f, classification_report(l_test, predicted)
print >>f, 'Accuracy: ', accuracy_score(l_test, predicted)*100.00,'%'
print >>f, 'Precision: ', precision_score(l_test, predicted, average = 'weighted')*100.00,'%'     
f.close()

# Decision Tree Classifier
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

model = pd.read_csv('~/Desktop/BigData/FinalProject/UFO_Modeling_Full.csv')
del model['Unnamed: 0']
model = model.reset_index(drop = True)

data = model[(model['Year']>=1900) & (model['Year']<=2015)]
data = data.reset_index(drop = True)
data['Shape'] = data['Shape'].replace('unspecified','other')
data = data.reindex(np.random.permutation(data.index)).reset_index(drop = True)

shape_counts = data.Shape.value_counts().tolist()
shape_labels = data.Shape.value_counts().index.tolist()
shape_percent = [shape_counts[i]*(100.0/data.Shape.value_counts().sum()) for i in xrange(len(shape_counts))]
f = open('Shape_Percentages_REV.txt','w')
for i in xrange(len(shape_percent)):
    print >>f, '%s ==>' % shape_labels[i],'%.2f' % shape_percent[i],'%'
f.close()

# Discrete variables
Timezone = pd.get_dummies(data['Timezone'])
Weekday = pd.get_dummies(data['Weekday'])
Month = pd.get_dummies(data['Month'])
Shape = pd.get_dummies(data['Shape'])
Color = pd.get_dummies(data['Color'])

# Excluding skewed light category? 
Timezone = pd.get_dummies(data.Timezone[(data.Shape != 'light')])
Weekday = pd.get_dummies(data.Weekday[(data.Shape != 'light')])
Month = pd.get_dummies(data.Month[(data.Shape != 'light')])
Shape = pd.get_dummies(data.Shape[(data.Shape != 'light')])
Color = pd.get_dummies(data.Color[(data.Shape != 'light')])

def norm(column):
    return np.abs((column-column.mean())/(column.max()-column.min()))

# Continuous variables
Longitude = norm(data.Lon)
Latitude = norm(data.Lat)
Witnesses = norm(data.Witnesses_ML)
Hour = norm(data.Hour)
Year = norm(data.Year)

features = pd.concat((data.Lat,data.Lon), axis = 1).as_matrix()
labels = data.Shape
f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size = 0.20)

RFC = RandomForestClassifier(criterion = 'entropy',
                             class_weight = 'balanced',
                             min_samples_split = 150,
                             random_state = 2)

#print(RFC)
RFC.fit(f_train,l_train)

# make RandomForestClassifier predictions
expected = l_test
predicted = RFC.predict(f_test)
predicted = pd.Series(predicted)
print "\n Precision: ", precision_score(expected, predicted, average = 'weighted')*100.00 ,"%"

# summarize the fit of the model to file
import datetime
x = [datetime.datetime.now().year,
     datetime.datetime.now().month,
     datetime.datetime.now().day,
     datetime.datetime.now().hour,
     datetime.datetime.now().minute,
     datetime.datetime.now().second]

f = open('RFC_Report%s.txt' % (''.join(str(s) for s in x)),'w')
print >>f, metrics.classification_report(expected, predicted)
print >>f, metrics.confusion_matrix(expected, predicted)
print >>f, 'Accuracy: ', accuracy_score(expected, predicted)*100.00,'%'
print >>f, 'Precision: ', precision_score(expected, predicted, average = 'weighted')*100.00,'%'
f.close()


# Plots trees to .dot files... Difficult to visualize still.
import pydot
from sklearn import tree
from IPython.display import Image 
from sklearn.externals.six import StringIO

# Make PDF's which show decisions being made for classifying shapes
# based on longitude and latitude values. Will need to install pydot for this.
i_tree = 0
for tree_in_forest in RFC.estimators_:
    with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
        dot_data = StringIO()
        tree.export_graphviz(tree_in_forest, out_file = dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("UFO_Tree_%s.pdf"%i_tree)
    i_tree = i_tree + 1
