#%%
#Imports
from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf     # version 1.15.2
tf.enable_eager_execution() #To fix error thrown

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statistics

#Used this LTSM tutorial often
#https://www.tensorflow.org/tutorials/structured_data/time_series#part_2_forecast_a_multivariate_time_series

#OBTAINED dataset from:
#https://www.ncdc.noaa.gov/cdo-web/search

#%%
#Create DataFrame
climateChangeDf = pd.read_csv('src/INPUT/TRAIN/NewOrleansTemperatures.csv').dropna() #dropna() to drop nans
print(climateChangeDf.head())

#%%
#Choosing features from data
#Using these three features to predict TAVG values
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

# %%
# function to create multivariate data array
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False): 
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size #Predicting 2015

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

#%%
# Setting history size and step size for our RNN
past_history = 120  #using data from past 10 years to make prediction
STEP = 1            #How big the step: 1 month


# %%
#Creating training and validation sets
# Potential target values:
# dataset[:, 0] == AverageTemperature, Using this one
# dataset[:, 1] == MaxTemperature
# dataset[:, 2] == MinTemperature

future_target = 36 #Predicting for next 3 year(s)

x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

# %%
#Printing single window of past history
print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))

# %%
#Shuffle, batch, and cache the dataset
BATCH_SIZE = 12     #How many rows are gone through before updating model ADD NAME
BUFFER_SIZE = 10000

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

# %%
#Function to plot multi-step model
def create_time_steps(length):
  return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()

# %%
#Plotting Example of Training data
for x, y in train_data_multi.take(1):
  multi_step_plot(x[0], y[0], np.array([0]))

# %%
#Creating multi-step RNN model with TWO LTSM layers
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(36)) #NEEDS to equal future target

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

# %%
#Training model
EVALUATION_INTERVAL = 58  #TRAIN_SPLIT #692//12 = 57.667
EPOCHS = 100               #How many times the model runs over the dataset

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=15) #VALIDATION_SPLIT #174//12 = 14.5

#%%
#SAVE Model
#Goes into folder MODEL
os.chdir('src/MODEL')

multi_step_model.save('NewOrleansAverageTemperatureModel.h5')  # creates a HDF5 file 'NewOrleansAverageTemperatureModel.h5'

#Returns to original directory
os.chdir('..')
os.chdir('..')

del multi_step_model  # deletes the existing model

#%%
#LOAD Model back in
climateModel = tf.keras.models.load_model('src/MODEL/NewOrleansAverageTemperatureModel.h5')

# %%
#Plotting model training and validation loss
def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

#Using function to plot model
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

# %%
#Plotting Accuracy of Model
for x, y in val_data_multi.take(3):
  multi_step_plot(x[0], y[0], climateModel.predict(x)[0])

# %%
#Extracting Actual and Predicted Values
for x, y in val_data_multi.take(1):
  print(x.shape)
  print(y.shape)
  historyValues = np.array(x)
  actualValues = np.array(y)
  predictedValues = np.array(climateModel.predict(x))


# %%
#function to unNormalize TAVG data
def unNormalize(data):
  data = [(x * landAvgTmp_std + landAvgTmp_mean) for x in data]
  return data

# %%
#unNormalizing Actual and Predicted Values
actualValues = unNormalize(actualValues)
predictedValues = unNormalize(predictedValues)

# %%
#Printing Actual and Predicted Values
print(actualValues)
print(predictedValues)

# %%
#Calculating Error Metrics
#Mean absolute error = 0 is perfect
mae = mean_absolute_error(actualValues, predictedValues)
#Mean squared error takes outliers into account, making it larger than Mean absolute error
mse = mean_squared_error(actualValues, predictedValues)
#R2 score = 1 is perfect
r2 = r2_score(actualValues, predictedValues)

print('Mean Absolute Error: {}'.format(mae))
print('Mean Squared Error: {}'.format(mse))
print('R2 Score: {}'.format(r2))

# %%
