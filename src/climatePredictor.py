#%%
#Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.enable_eager_execution() #To fix error thrown

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

#%%
#Create DataFrame
climateChangeDf = pd.read_csv('GlobalTemperatures.csv').dropna() #dropna() to drop nans
print(climateChangeDf.head())

#%%
#Choosing features from data
features_considered = ['LandAverageTemperature', 'LandAverageTemperatureUncertainty']
features = climateChangeDf[features_considered]
features.index = climateChangeDf['dt']
print(features.head())
print(features.tail())

#%%
#Plotting Data
features.plot(subplots=True)

# %%
# Calculating mean and standard deviation
TRAIN_SPLIT = 1593 #Around 80% of rows
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std

# %%
# function to create multivariate data array
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

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
past_history = 240 #using data from past 20 years
future_target = 60 #predicting for next 5 years
STEP = 12           #How big the step: 1 year

#Creating training data array
x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
#Creating validation array 
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1], 
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True) #Validation data

# %%
#View one data point from training set
print ('Single window of past history : {}'.format(x_train_single[0].shape))

# %%
#shuffle, batch, and cache the dataset
BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()


# %%
#Using Long Short Term Memory (LTSM) in Recurrent Neural Network (RNN) to create model
single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32,
                                           input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')


# %%
# Sample prediction
for x, y in val_data_single.take(1):
  print(single_step_model.predict(x).shape)

# %%
# Training Model
EVALUATION_INTERVAL = 200 #Each epoch will run for 200 steps LIKELY WILL ADJUST
EPOCHS = 10
single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)


# %%
#Function to produce predictions using LTSM
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


# %%
#Plotting single-step training
plot_train_history(single_step_history,
                   'Single Step Training and validation loss')

# %%
#Functions to create time steps and show plots
#Function to create time steps
def create_time_steps(length):
  return list(range(-length, 0))

#Function to show plots
def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

# %%
#Plotting simple LTSM model
for x, y in val_data_single.take(3):
  plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                    single_step_model.predict(x)[0]], 12,
                   'Single Step Prediction')
  plot.show()

# %%
#Will work on Multi-step model