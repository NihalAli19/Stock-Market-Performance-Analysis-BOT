import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

df = pd.read_csv("TSLA.csv")
df["Date"] = pd.to_datetime(df["Date"])

fig = go.Figure(data=[go.Candlestick(x=df["Date"],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'])])
fig.show()

data = df.dropna()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_Data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

timetopredict = 90

xtrain = []
ytrain = []

for x in range(timetopredict, len(scaled_Data)):
	xtrain.append(scaled_Data[x - timetopredict:x, 0])
	ytrain.append(scaled_Data[x, 0])

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(xtrain, ytrain, epochs=25, batch_size=32)

test_start_date = data["Date"].iloc[-timetopredict]
test_data = data[data["Date"] >= test_start_date]
test_data = pd.merge(data["Close"], test_data)

actual_test_price = test_data["Close"].values

total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)

input_models = total_dataset[len(total_dataset) - len(test_data) - timetopredict:].values
input_models = input_models.reshape((-1, 1))
input_models = scaler.transform(input_models)



xtest = []

for x in range(timetopredict, len(input_models)):
	xtest.append(input_models[x - timetopredict:x, 0])

xtest = np.array(xtest)
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

testprice = model.predict(xtest)

testprice = scaler.inverse_transform(testprice)

plt.plot(actual_test_price,label = "real")
plt.plot(testprice,label = "predict")
plt.legend()
plt.show()



