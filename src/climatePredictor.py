#%%
#Imports

import tensorflow as tf # version 1.15.2

#%%
#LOAD Model In
climateModel = tf.keras.models.load_model('landAverageTemperatureModel.h5')

# %%
#Testing
prediction = climateModel.predict(x)[0]
print(prediction)

# %%
