#%%
#Imports

import tensorflow as tf # version 1.15.2
import numpy as np
import pandas as pd

#%%
#Create DataFrame
#dropna() made df start on 1850-1-1 (row 1202), may change
climateChangeDf = pd.read_csv('INPUT/TRAIN/GlobalTemperatures.csv').dropna() #dropna() to drop nans
print(climateChangeDf.head())

#%%
#Choosing features from data
#Using these three features to predict future LandAverageTemperature values
features_considered = ['LandAverageTemperature', 'LandMaxTemperature', 'LandMinTemperature']
features = climateChangeDf[features_considered]
features.index = climateChangeDf['dt']
print(features)

#%%
#Plotting Data
features.plot(subplots=True)

# %%
# Calculating mean and standard deviation. Pre-processing data
# 1992 rows
TRAIN_SPLIT = 1593 #Around 80% of rows 

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0) #Goes to row 2795 (10-1-1982)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

#WILL USE to unNormalize the LandAverageTemperature prediction
landAvgTmp_mean = statistics.mean([x[0] for x in dataset[:TRAIN_SPLIT]])
landAvgTmp_std = statistics.stdev([x[0] for x in dataset[:TRAIN_SPLIT]])

#Normalizing data
#Important for Neural Networks
dataset = (dataset-data_mean)/data_std

#%%
#LOAD Model In
climateModel = tf.keras.models.load_model('MODEL/landAverageTemperatureModel.h5')

# %%
#function to unNormalize LandAverageTemp data
def unNormalize(data):
  data = [(x * landAvgTmp_std + landAvgTmp_mean) for x in data]
  return data

# %%
#Predicting LandAvgTemp for 2016 (UNKNOWN VALUES)
#Retrieving last 10 years of data
history2005_2015Values = dataset[-120:]

#Expanding dimension so the RNN can read it
history2005_2015Values = np.expand_dims(history2005_2015Values, axis=0)
print(history2005_2015Values.shape)

#Feeding data to RNN
predictedLandAvgTemp2016 = climateModel.predict(history2005_2015Values)

# %%
#UnNormalizing prediction for 2016
predictedLandAvgTemp2016 = unNormalize(predictedLandAvgTemp2016)
print(predictedLandAvgTemp2016)

# %%
#Printing Predicted Land Average Temperatures for 2016
print('Predicted Land Average Temperature for 2016: \n')

print('January: {}'.format(predictedLandAvgTemp2016[0][0]))
print('February: {}'.format(predictedLandAvgTemp2016[0][1]))
print('March: {}'.format(predictedLandAvgTemp2016[0][2]))
print('April: {}'.format(predictedLandAvgTemp2016[0][3]))
print('May: {}'.format(predictedLandAvgTemp2016[0][4]))
print('June: {}'.format(predictedLandAvgTemp2016[0][5]))
print('July: {}'.format(predictedLandAvgTemp2016[0][6]))
print('August: {}'.format(predictedLandAvgTemp2016[0][7]))
print('September: {}'.format(predictedLandAvgTemp2016[0][8]))
print('October: {}'.format(predictedLandAvgTemp2016[0][9]))
print('November: {}'.format(predictedLandAvgTemp2016[0][10]))
print('December: {}'.format(predictedLandAvgTemp2016[0][11]))

