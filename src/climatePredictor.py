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

# %%
#Saving Test Results
prediction_dict = {'January':[predictedTAVG[0][0], predictedTAVG[0][12], predictedTAVG[0][24]],\
            'February':[predictedTAVG[0][1], predictedTAVG[0][13], predictedTAVG[0][25]],\
            'March':[predictedTAVG[0][2], predictedTAVG[0][14], predictedTAVG[0][26]],\
            'April':[predictedTAVG[0][3], predictedTAVG[0][15], predictedTAVG[0][27]],\
            'May':[predictedTAVG[0][4], predictedTAVG[0][16], predictedTAVG[0][28]],\
            'June':[predictedTAVG[0][5], predictedTAVG[0][17], predictedTAVG[0][29]],\
            'July':[predictedTAVG[0][6], predictedTAVG[0][18], predictedTAVG[0][30]],\
            'August':[predictedTAVG[0][7], predictedTAVG[0][19], predictedTAVG[0][31]],\
            'September':[predictedTAVG[0][8], predictedTAVG[0][20], predictedTAVG[0][32]],\
            'October':[predictedTAVG[0][9], predictedTAVG[0][21], predictedTAVG[0][33]],\
            'November':[predictedTAVG[0][10], predictedTAVG[0][22], predictedTAVG[0][34]],\
            'December':[predictedTAVG[0][11], predictedTAVG[0][23], predictedTAVG[0][35]]} 
  
testResults = pd.DataFrame(prediction_dict, index =['2020', '2021', '2022']) 

#Printing Predictions  
print(testResults)

testResults.to_csv('OUTPUT/NewOrleansTAVGPredictions.csv', index = True, header=True)
# %%
