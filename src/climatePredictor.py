#%%
#Imports
import tensorflow as tf # version 1.15.2
import numpy as np
import pandas as pd
import statistics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

#%%
#Create DataFrame
climateChangeDf = pd.read_csv('INPUT/TRAIN/NewOrleansTemperatures.csv').dropna() #dropna() to drop nans

#%%
#Calculations to unNormalize prediction
features_considered = ['TAVG', 'TMAX', 'TMIN']
features = climateChangeDf[features_considered]
features.index = climateChangeDf['DATE']

# 866 rows
TRAIN_SPLIT = 692 #Around 80% of rows 

#WILL USE to normalize test data
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0) 
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

#WILL USE to unNormalize the TAVG prediction
landAvgTmp_mean = statistics.mean([x[0] for x in dataset[:TRAIN_SPLIT]])
landAvgTmp_std = statistics.stdev([x[0] for x in dataset[:TRAIN_SPLIT]])

#%%
#Saving test data
newOrleansTemp2010_2019df = climateChangeDf[-122: -2]
newOrleansTemp2010_2019df.to_csv (r'INPUT/TEST/NewOrleansTemperatures2010-2019.csv', index = False, header=True)

#%%
#Read test data back in
testDataDf = pd.read_csv('INPUT/TEST/NewOrleansTemperatures2010-2019.csv')

#%%
#Choosing features from data
#Using these three features to predict future TAVG values
testFeatures_considered = ['TAVG', 'TMAX', 'TMIN']
testFeatures = testDataDf[features_considered]
testFeatures.index = testDataDf['DATE']
print(testFeatures)

#%%
#Plotting Data

#Converting date string into real datetime objects
dates = mdates.num2date(mdates.datestr2num(testFeatures.index))

fig, axs = plt.subplots(3)
axs[0].plot(dates, testFeatures['TMAX'], color = 'red')
axs[0].set_title('New Orleans Maximum Temperature')
axs[0].set_ylabel('Temperature {}'.format(u'\u2103'))

axs[1].plot(dates, testFeatures['TAVG'], color = 'purple')
axs[1].set_title('New Orleans Average Temperature')
axs[1].set_ylabel('Temperature {}'.format(u'\u2103'))

axs[2].plot(dates, testFeatures['TMIN'], color = 'blue')
axs[2].set_title('New Orleans Minimum Temperature')
axs[2].set_ylabel('Temperature {}'.format(u'\u2103'))

fig.tight_layout()
plt.show()

# %%
# Pre-processing data
testDataset = testFeatures.values

#Normalizing data
#Important for Neural Networks
testDataset = (testDataset-data_mean)/data_std

#%%
#LOAD Model In
climateModel = tf.keras.models.load_model('MODEL/NewOrleansAverageTemperatureModel.h5')

# %%
#function to unNormalize LandAverageTemp data
def unNormalize(data):
  data = [(x * landAvgTmp_std + landAvgTmp_mean) for x in data]
  return data

# %%
#Predicting LandAvgTemp for 2020 (UNKNOWN VALUES)

#Expanding dimension so the RNN can read it
testDataset = np.expand_dims(testDataset, axis=0)
print(testDataset.shape)

#Feeding data to RNN
predictedTAVG = climateModel.predict(testDataset)

# %%
#UnNormalizing predictions
predictedTAVG = unNormalize(predictedTAVG)

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
