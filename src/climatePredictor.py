#%%
#Imports

import tensorflow as tf # version 1.15.2
import numpy as np
import pandas as pd
import statistics
import os

#%%
#Create DataFrame
climateChangeDf = pd.read_csv('src/INPUT/TRAIN/NewOrleansTemperatures.csv').dropna() #dropna() to drop nans

#%%
#Choosing features from data
#Using these three features to predict future TAVG values
features_considered = ['TAVG', 'TMAX', 'TMIN']
features = climateChangeDf[features_considered]
features.index = climateChangeDf['DATE']
print(features)

#%%
#Plotting Data
features.plot(subplots=True)

# %%
# Calculating mean and standard deviation. Pre-processing data
# 866 rows
TRAIN_SPLIT = 692 #Around 80% of rows 

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0) 
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

#WILL USE to unNormalize the TAVG prediction
landAvgTmp_mean = statistics.mean([x[0] for x in dataset[:TRAIN_SPLIT]])
landAvgTmp_std = statistics.stdev([x[0] for x in dataset[:TRAIN_SPLIT]])

#Normalizing data
#Important for Neural Networks
dataset = (dataset-data_mean)/data_std

#%%
#LOAD Model In
climateModel = tf.keras.models.load_model('src/MODEL/NewOrleansAverageTemperatureModel.h5')

# %%
#function to unNormalize LandAverageTemp data
def unNormalize(data):
  data = [(x * landAvgTmp_std + landAvgTmp_mean) for x in data]
  return data

# %%
#Predicting LandAvgTemp for 2020 (UNKNOWN VALUES)
#Retrieving last 10 years of data (2009-2019)
history2009_2019Values = dataset[-122:-2]

#%%
#Saving test data
#Goes into INPUT/TEST folder
os.chdir('src/INPUT/TEST')
print(os.getcwd())
'''
with open("%s.csv" %searchKey, "w", newline = '') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerow(studentData)
os.chdir(path)
'''
#Returns to original directoy
os.chdir('..')
os.chdir('..')
os.chdir('..')
print(os.getcwd())

#%%
#Expanding dimension so the RNN can read it
history2009_2019Values = np.expand_dims(history2009_2019Values, axis=0)
print(history2009_2019Values.shape)

#Feeding data to RNN
predictedTAVG = climateModel.predict(history2009_2019Values)

# %%
#UnNormalizing predictions
predictedTAVG = unNormalize(predictedTAVG)
print(predictedTAVG)

#%%
#function to print monthly temperatures
def printMonthTemps(year, predictions):
  print('Predicted Land Average Temperature for {}: \n'.format(year))

  print('January: {}'.format(predictions[0]))
  print('February: {}'.format(predictions[1]))
  print('March: {}'.format(predictions[2]))
  print('April: {}'.format(predictions[3]))
  print('May: {}'.format(predictions[4]))
  print('June: {}'.format(predictions[5]))
  print('July: {}'.format(predictions[6]))
  print('August: {}'.format(predictions[7]))
  print('September: {}'.format(predictions[8]))
  print('October: {}'.format(predictions[9]))
  print('November: {}'.format(predictions[10]))
  print('December: {}'.format(predictions[11]))

  print('\n\n')

#%%
#Printing functions for 2020, 2021, and 2022
printMonthTemps(2020, predictedTAVG[0][:12])
printMonthTemps(2021, predictedTAVG[0][12:24])
printMonthTemps(2022, predictedTAVG[0][24:36])

# %%
