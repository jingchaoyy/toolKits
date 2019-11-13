"""
Created on  11/13/2019
@author: Jingchao Yang

https://www.kaggle.com/msripooja/hourly-energy-consumption-time-series-rnn-lstm
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.metrics import r2_score

from keras.layers import Dense, Dropout, SimpleRNN, LSTM
from keras.models import Sequential


def normalize_data(df):
    scaler = sklearn.preprocessing.MinMaxScaler()
    df['DOM_MW'] = scaler.fit_transform(df['DOM_MW'].values.reshape(-1, 1))
    return df


def load_data(stock, seq_len):
    X_train = []
    y_train = []
    for i in range(seq_len, len(stock)):
        X_train.append(stock.iloc[i - seq_len: i, 0])
        y_train.append(stock.iloc[i, 0])

    # 1 last 6189 days are going to be used in test
    X_test = X_train[110000:]
    y_test = y_train[110000:]

    # 2 first 110000 days are going to be used in training
    X_train = X_train[:110000]
    y_train = y_train[:110000]

    # 3 convert to numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 4 reshape data to input into RNN models
    X_train = np.reshape(X_train, (110000, seq_len, 1))

    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))

    return [X_train, y_train, X_test, y_test]


def plot_predictions(test, predicted, title):
    plt.figure(figsize=(16, 4))
    plt.plot(test, color='blue', label='Actual power consumption data')
    plt.plot(predicted, alpha=0.7, color='orange', label='Predicted power consumption data')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized power consumption scale')
    plt.legend()
    plt.show()


# check all the files in the input dataset
print(os.listdir("sampleData"))

# choosing DOM_hourly.csv data for analysis
fpath = 'sampleData/DOM_hourly.csv'

df = pd.read_csv(fpath, index_col='Datetime', parse_dates=['Datetime'])
# checking missing data
print('missing data', df.isna().sum())

df_norm = normalize_data(df)
df_norm.shape

# create train, test data
seq_len = 20  # choose sequence length

X_train, y_train, X_test, y_test = load_data(df, seq_len)

print('X_train.shape = ', X_train.shape)
print('y_train.shape = ', y_train.shape)
print('X_test.shape = ', X_test.shape)
print('y_test.shape = ', y_test.shape)

lstm_model = Sequential()

lstm_model.add(LSTM(40, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(40, activation="tanh", return_sequences=True))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(40, activation="tanh", return_sequences=False))
lstm_model.add(Dropout(0.15))

lstm_model.add(Dense(1))

lstm_model.summary()

'''Let's check r2 score for the values predicted by the above trained LSTM model'''
lstm_predictions = lstm_model.predict(X_test)

lstm_score = r2_score(y_test, lstm_predictions)
print("R^2 Score of LSTM model = ", lstm_score)

'''Let's compare the actual values vs predicted values by plotting a graph'''
plot_predictions(y_test, lstm_predictions, "Predictions made by LSTM model")
