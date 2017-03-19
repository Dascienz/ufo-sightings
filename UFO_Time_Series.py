# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 18:30:18 2016

@author: Dascienz
"""

"""
1.) Time-series plots by decade

2.) RNN for fitting/predicting sightings counts over time.
"""

# TIME-SERIES DATA
import pandas as pd
import numpy as np

model = pd.read_csv('~/Desktop/BigData/FinalProject/UFO_Modeling_Full.csv')
del model['Unnamed: 0']
model = model.reset_index(drop = True)
model['Date'] = model['Date'].astype('datetime64')
time_series = pd.DataFrame({'Date':model.Date.value_counts().sort_index().index,
                            'Sightings':model.Date.value_counts().sort_index().values})

# The 1950's! Made in the shade!
plt.figure()
plt.title("No. Sightings: 1950's")
plt.plot(time_series.Date,time_series.Sightings)
plt.xlim([pd.Timestamp('1950-01-01'), pd.Timestamp('1960-01-01')])
plt.ylim([1,15])
plt.xlabel('Year')
plt.ylabel('No. Sightings')
plt.xticks(rotation='45')
plt.grid(True)
#plt.savefig('UFO_Time_Series_1950s.png',format='png',dpi=300)
plt.show()


# The 1960's! Groovy!
plt.figure()
plt.title("No. Sightings: 1960's")
plt.plot(time_series.Date,time_series.Sightings)
plt.xlim([pd.Timestamp('1960-01-01'), pd.Timestamp('1970-01-01')])
plt.ylim([1,30])
plt.xlabel('Year')
plt.ylabel('No. Sightings')
plt.xticks(rotation='45')
plt.grid(True)
#plt.savefig('UFO_Time_Series_1960s.png',format='png',dpi=300)
plt.show()

# The 1970's! Can you dig it?
plt.figure()
plt.title("No. Sightings: 1970's")
plt.plot(time_series.Date,time_series.Sightings)
plt.xlim([pd.Timestamp('1970-01-01'), pd.Timestamp('1980-01-01')])
plt.ylim([1,30])
plt.xlabel('Year')
plt.ylabel('No. Sightings')
plt.xticks(rotation='45')
plt.grid(True)
#plt.savefig('UFO_Time_Series_1970s.png',format='png',dpi=300)
plt.show()

# The 1980's! Tubular!
plt.figure()
plt.title("No. Sightings: 1980's")
plt.plot(time_series.Date,time_series.Sightings)
plt.xlim([pd.Timestamp('1980-01-01'), pd.Timestamp('1990-01-01')])
plt.ylim([1,30])
plt.xlabel('Year')
plt.ylabel('No. Sightings')
plt.xticks(rotation='45')
plt.grid(True)
#plt.savefig('UFO_Time_Series_1980s.png',format='png',dpi=300)
plt.show()

# The 1990's! All that and a bag of chips!
plt.figure()
plt.title("No. Sightings: 1990's")
plt.plot(time_series.Date,time_series.Sightings)
plt.xlim([pd.Timestamp('1990-01-01'), pd.Timestamp('2000-01-01')])
plt.ylim([1,60])
plt.xlabel('Year')
plt.ylabel('No. Sightings')
plt.xticks(rotation='45')
plt.grid(True)
#plt.savefig('UFO_Time_Series_1990s.png',format='png',dpi=300)
plt.show()

# THe 2000's! The distant future, the year 2000!
plt.figure()
plt.title("No. Sightings: 2000's")
plt.plot(time_series.Date,time_series.Sightings)
plt.xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2016-01-01')])
plt.ylim([1,150])
plt.xlabel('Year')
plt.ylabel('No. Sightings')
plt.xticks(rotation='45')
plt.grid(True)
#plt.savefig('UFO_Time_Series_2000s.png',format='png',dpi=300)
plt.show()

# Define time flow as days
time = time_series.values

time_delta = []
for i in range(len(time[:,0])):
    time_delta.append(time[:,0][i].date()-time[:,0][0].date())

time_delta = map(lambda t: t.days, time_delta)
time_delta = map(lambda t: t/365., time_delta)

# Time-series prediction? Seems fairly periodic for 3 decades
# until ~1995! Wonder what we can find out.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as dates
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Recurrent Neural Network for Time-Series Predictions
# Define time flow as days
import pandas as pd
import numpy as np
model = pd.read_csv('~/Desktop/BigData/FinalProject/UFO_Modeling_Full.csv')
del model['Unnamed: 0']
model = model.reset_index(drop = True)
model['Date'] = model['Date'].astype('datetime64')

time_series = pd.DataFrame({'Date':model.Date.value_counts().sort_index().index,
                            'Sightings':model.Date.value_counts().sort_index().values})

time = time_series[(time_series.Date >= pd.Timestamp('1960-01-01'))].values

np.random.seed(9)
df = pd.DataFrame({'Sightings': time[:,1]})
time_data = df.values
time_data = time_data.astype('float32')  

train_size = int(len(time_data) * 0.70)
test_size = len(time_data) - train_size
train = time_data[0:train_size,:] 
test = time_data[train_size:len(time_data),:]

# Time step function: X: values at (t) -> Y: values at (t + 1)           
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Create and fit Multilayer Perceptron model
# Needs work, since RMSE still not minimized to a decent extent.
from keras.layers.core import Dense, Activation, Dropout
model_t = Sequential()
model_t.add(Dense(16, input_dim = 1, activation='relu'))
model_t.add(Dense(1))
model_t.compile(loss='mse', optimizer='adam')
model_t.fit(trainX, trainY, nb_epoch=300, batch_size=32, verbose=2)

# Estimate model performance
# MSE stands for mean squared error.
trainScore = model_t.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))
testScore = model_t.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))

# Generate predictions for training
trainPredict = model_t.predict(trainX)
testPredict = model_t.predict(testX)

# Shift train predictions for plotting
trainPredictPlot = np.empty_like(time_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(time_data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(time_data)-1, :] = testPredict

# Plot baseline and predictions
plt.figure(figsize=(8,6))
plt.title('Time-Series Predictions')
plt.plot(time_data, color='black')
plt.plot(trainPredictPlot, color = 'blue')
plt.plot(testPredictPlot, color = 'red')
plt.legend(['Data','Train','Test'], loc = 'best')
plt.axis('tight')
plt.xlim([6400,8200])
plt.ylim([1,100])
plt.xlabel('Index Number')
plt.ylabel('No. Sightings')
plt.grid(True)
#plt.savefig('Time_Predictions_22.70RMSE.png',format='png',dpi=300)
plt.show()

