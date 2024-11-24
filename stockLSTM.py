import urllib.request, json
import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import dataForLSTM

#Tensorflow with keras is used to implement the (stacked) LSTM model
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM

#apiKey = "QCLLY08TISZBMXL9"
ticker = "BAC"

#JSON file with daily time series data (date, daily open, daily high, daily low, daily close, daily volume) of specified company,
#covering 20 years
urlString = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=QCLLY08TISZBMXL9"%ticker

#Will use the close values only for the prediction, hence need to save the dates (important as we are using a time series model)
#and the close values.

#Save data to this file
file = "stock_market_data-%s.csv"%ticker

#If the data has not been saved, save the data to the file and store it as a Pandas dataframe
if not os.path.exists(file):
    with urllib.request.urlopen(urlString) as url:
        data = json.loads(url.read().decode())
        data = data["Time Series (Daily)"]
        df = pd.DataFrame(columns = ["Date", "Close"])
        for k,v in data.items():
            date = dt.datetime.strptime(k, "%Y-%m-%d")
            dataRow = [date.date(), float(v["4. close"])]
            df.loc[-1,:] = dataRow
            df.index = df.index + 1
    print("Data saved to : %s"%file)        
    df.to_csv(file)

# If the data has already been saved, load it to a Pandas dataframe
else:
    print("File already exists. Loading data from CSV")
    df = pd.read_csv(file)

#Sorts the dataframe by date (chronologically)
df = df.sort_values("Date")

#Stores close values
close = df[["Close"]]
dfClose = close.values

#Uses MinMaxScaler to scale the close values so that the are between 0 and 1, and reshapes the resulting numpy array as an array with
#one column (and as many rows as necessary).
scaler = MinMaxScaler((0, 1))
dfScaled = scaler.fit_transform(np.array(dfClose).reshape(-1, 1))

#Splits the scaled close values into a training dataset and testing dataset. 90% of the data is used for training, the rest
#allocated for testing. 
trainSize = int(len(dfScaled) * 0.9)
testSize = len(dfScaled) - trainSize
dfTrain = dfScaled[0:trainSize,:]
dfTest = dfScaled[trainSize:len(dfScaled),:1]

#Will use the previous n days to predict the (n+1)st day, where n is the number of previous days. This function creates two numpy
#arrays to represent this, where each item in X stores n sequential days and each corresponding item in Y stores the following day.
def createDF(data, n):
    X = []
    Y = []
    for i in range(len(data) - n - 1):
        X.append(data[i:(i + n), 0])
        Y.append(data[i + n, 0])
    return np.array(X), np.array(Y)

#Will use 50 previous days to predict the next day
numberOfPreviousDays = 50

#Uses function to create the X and Y arrays for the training and testing datasets.
Xtrain, Ytrain = createDF(dfTrain, numberOfPreviousDays)
Xtest, Ytest = createDF(dfTest, numberOfPreviousDays)

#Reshapes the two X arrays so that they are in the required form to use in the LSTM model [batch size, number of timesteps, and
#the 'feature']. In this case feature = 1, since we are using univariate forecasting.
Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 1)
Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 1)

#Creates a stacked LSTM model with three LSTM layers and a dense layer
model = Sequential()

#First two LSTM layers have 64 neurons, and return the full output sequence
model.add(LSTM(units = 64, return_sequences = True, input_shape = (Xtrain.shape[1], 1)))
model.add(LSTM(units = 64, return_sequences = True))

#Last LSTM layer also has 64 neurons, but returns the last value in the output sequence only (want to predict one day at a time)
model.add(LSTM(units = 64))

#Dense layer with one neuron and linear activiation function
model.add(Dense(units = 1, activation = "linear"))

model.summary()

#Configures the losses as the mean squared error and the optimiser as the Adam optimiser
model.compile(loss = "mean_squared_error", optimizer = "adam")

#Trains the data on the Xtrain and Ytrain arrays, over 100 epochs and trains 64 samples at a time
model.fit(Xtrain, Ytrain, validation_data = (Xtest, Ytest), epochs = 100, batch_size = 64)

#Saves model to a JSON file
modelFile = model.to_json()
with open("model%s.json"%ticker, "w") as jsonFile:
    jsonFile.write(modelFile)
model.save_weights("model%s.h5"%ticker)
print("Saved model")