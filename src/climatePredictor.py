#%%
#Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf # version 1.15.2
tf.enable_eager_execution() #To fix error thrown

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

#Used this LTSM tutorial often
#https://www.tensorflow.org/tutorials/structured_data/time_series#part_2_forecast_a_multivariate_time_series

#%%
#Create DataFrame
#dropna() made df start on 1850-1-1 (row 1202), may change
climateChangeDf = pd.read_csv('GlobalTemperatures.csv').dropna() #dropna() to drop nans
print(climateChangeDf.head())

#%%
#Choosing features from data
features_considered = ['LandAverageTemperature', 'LandMaxTemperature', 'LandMinTemperature']
features = climateChangeDf[features_considered]
features.index = climateChangeDf['dt']
print(features)

#%%
#Plotting Data
features.plot(subplots=True)

# %%
# Calculating mean and standard deviation
# 1992 rows
TRAIN_SPLIT = 1593 #Around 80% of rows 

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0) #Goes to row 2795 (10-1-1982)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

#Normalizing data
dataset = (dataset-data_mean)/data_std

# %%
# function to create multivariate data array
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size #Predicting from 2010 to 2015

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

#%%
# Setting parameters for our dataset
past_history = 120 #using data from past 10 years (since 2015) there is a LIMIT to this
STEP = 1           #How big the step: 1 month


# %%
#Creating training and validation sets
future_target = 12 #Predicting for next 1 year(s)
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

# %%
#Printing single window of past history
print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))

# %%
#Shuffle, batch, and cache the dataset
BATCH_SIZE = 256 #How many rows are gone through before updating model (WILL UPDATE)
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
#Plotting averageTemperature data
for x, y in train_data_multi.take(1):
  multi_step_plot(x[0], y[0], np.array([0]))

# %%
#Creating multi-step model using LTSM in RNN
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(12)) #NEEDS to equal future target

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

# %%
#Training model
EVALUATION_INTERVAL = 200 #TRAIN_SPLIT #1593
EPOCHS = 10 #How many times the model runs over the dataset

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

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
#Plotting multi-step model for all features
for x, y in val_data_multi.take(3):
  multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])

# %%
#Testing
for x, y in val_data_multi.take(3):
  print(x)
'''
for x, y in val_data_multi.take(1):
  actualValues = np.array(y[0])
  predictedValues = np.array(multi_step_model.predict(x)[0])

print(actualValues)
print(predictedValues)
predictedValues = multi_step_model.predict(x)


#Mean Absolute Error calculation
#mae = accuracy_score(actualValues, predictedValues)
#print(accuracy)
'''

# %%
