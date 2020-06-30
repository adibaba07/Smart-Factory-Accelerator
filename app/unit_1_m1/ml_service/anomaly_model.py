from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from tensorflow.random import set_seed
from sklearn.externals import joblib
import h5py
import numpy as np
import pandas as pd
import json
# import joblib
import logging
import tensorflow as tf
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt # Plotting library
set_seed(10)
sns.set(color_codes=True)

# %matplotlib inline - unique to jupyter notebook, won't work for python syntax: use plt.show() instead
# tf.logging.set_verbosity(tf.logging.ERROR) - deprecated

# create Logs of functionalities
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

data = pd.read_csv('../data/Averaged_BearingTest_Dataset.csv', index_col=[0])
# print(data.head())

# set seed
seed(10)

# split data into train and test
train = data['2004-02-12 10:52:39': '2004-02-15 22:52:39']  # normal conditions
test = data['2004-02-15 22:52:39':]  # data leading to failure

# print(train.shape)  # (505, 4)
# print(test.shape)  # (478, 4)

# seaborn plots

# fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
# ax = sns.lineplot(data=train['Bearing 1'], markers=True, hue='Bearing 1')
# ax = sns.lineplot(data=train['Bearing 2'], markers=True, hue='Bearing 2')
# ax = sns.lineplot(data=train['Bearing 3'], markers=True, hue='Bearing 3')
# ax = sns.lineplot(data=train['Bearing 4'], markers=True, hue='Bearing 4')
# plt.legend(loc='lower left')
# plt.show()

# normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
scaler_filename = "../ml_service/scaler.joblib"
joblib.dump(scaler, scaler_filename)

# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
# print("Training data shape:", X_train.shape)
# (505,1,4)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
# print("Test data shape:", X_test.shape)
# (478,1,4)

# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True,
              kernel_regularizer=regularizers.l2(0.00))(inputs)  # return_sequences=True ->each cell-inputs emits output
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)  # return_sequences=False ->last cell-input emits output
    L3 = RepeatVector(X.shape[1])(L2)  # duplicates embedded features - timesteps X features array for input in decoder
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)
    model1 = Model(inputs=inputs, outputs=output)
    return model1


# create the auto-encoder model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
# print(model.summary())

# fit the model to the data
nb_epochs = 100
batch_size = 10
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.05).history
scores = model.evaluate(X_train, X_train, verbose=0)
# print(scores, model.metrics_names)

# serialize model to JSON
# model_json = model.to_json()
# with open("../ml_service/model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("../ml_service/model.h5")
# print("Saved model to disk")

# print(np.mean(history['loss']))
# print(np.mean(history['val_loss']))

# Training Loss plot - to check if model is (overfitting - if val_loss very much greater than training loss)
# Here, avg Training loss = 0.0928 - check plot for comparison of losses

# fig, ax = plt.subplots(sharey='all', sharex='all')
# ax.plot(history['loss'], 'b', label='Train', linewidth=2)
# ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
# ax.set_title('Model Loss')
# ax.set_ylabel('Loss MAE')
# ax.set_xlabel('Epoch')
# ax.legend(loc='upper right')
# plt.show()

# plot loss distribution of training set to identify suitable threshold to be used for tagging anomalies
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
# X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[1])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
# print(X_pred.head())
X_pred.index = train.index

scored = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
# Xtrain = X_train.reshape(X_train.shape[0],X_train.shape[1])
# print(Xtrain)
scored['Loss_MAE'] = np.mean(np.abs(X_pred-Xtrain), axis=1)
# print(scored.head())

# plt.figure(figsize=(16,9), dpi=80)
# plt.title('Loss Distribution')
# sns.distplot(scored['Loss_MAE'], bins=20, kde=True, color='blue')
# plt.xlim([0.0, .5])
# plt.show()

# Let loss threshold be 0.25 based on above plot
# Calculate reconstruction loss on test set
X_pred = model.predict(X_test)
# print(X_pred)
# print(np.mean(sqrt(mean_squared_error(X_test,X_pred))))
# print(np.mean(np.abs(X_pred-X_test)))
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
# print(X_pred.head())
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
# print(sqrt(mean_squared_error(X_test,X_pred)))
scored['Loss_MAE'] = np.mean(np.abs(X_pred-Xtest), axis=1)
# scored['RMSE'] = sqrt(mean_squared_error(Xtest,X_pred))
scored['threshold'] = 0.25
scored['Anomaly'] = scored['Loss_MAE'] > scored['threshold']
print(scored.head())
# plt.plot(scored.index, scored.Loss_MAE, label = 'Loss_MAE')
# plt.plot(scored.index, scored.threshold, label = 'threshold')
# plt.legend()
# plt.show()

anomalies = scored[scored.Anomaly == True]
# print(anomalies.head())

# calculate the same metrics for the training set
# and merge all data in a single dataframe

# X_pred_train = model.predict(X_train)
# X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
# X_pred_train = pd.DataFrame(X_pred_train, columns=train.columns)
# X_pred_train.index = train.index
#
# scored_train = pd.DataFrame(index=train.index)
# scored_train['Loss_MAE'] = np.mean(np.abs(X_pred_train - Xtrain), axis=1)
# scored_train['threshold'] = 0.25
# scored_train['Anomaly'] = scored_train['Loss_MAE'] > scored_train['threshold']
# scored = pd.concat([scored_train, scored])

# bearing failures plot
# scored.plot(logy=True, color=['blue', 'red'])
# plt.show()

# model.save("../ml_service/model1.h5")
joblib.dump(model, "../ml_service/model.joblib")
print("Model saved")
