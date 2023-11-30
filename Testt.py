# Import necessary libraries
import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from datetime import timedelta

# Read the CSV file containing stock data into a DataFrame
df = pd.read_csv("TSLA.csv")

# Convert the "Date" column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Create a candlestick chart using Plotly
fig = go.Figure(data=[go.Candlestick(x=df["Date"],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'])])
fig.show()

# Drop rows with missing values from the DataFrame
data = df.dropna()

# Scale (normalize) the "Close" prices to a range between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_Data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

# Number of previous days to consider for prediction
timetopredict = 90

# Prepare training data for LSTM model
xtrain = []
ytrain = []
for x in range(timetopredict, len(scaled_Data)):
    xtrain.append(scaled_Data[x - timetopredict:x, 0])
    ytrain.append(scaled_Data[x, 0])

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

# Create a Sequential model with LSTM layers for time series prediction
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the LSTM model with training data
model.fit(xtrain, ytrain, epochs=25, batch_size=32)

# Select the start date for the test data
test_start_date = data["Date"].iloc[-timetopredict]

# Extract test data from the selected start date onwards
test_data = data[data["Date"] >= test_start_date]
test_data = pd.merge(data["Close"], test_data)

# Get the actual test prices for comparison
actual_test_price = test_data["Close"].values

# Combine training and test data for future prediction
total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)

# Prepare input data for the LSTM model from the test data
input_models = total_dataset[len(total_dataset) - len(test_data) - timetopredict:].values
input_models = input_models.reshape((-1, 1))
input_models = scaler.transform(input_models)

xtest = []
for x in range(timetopredict, len(input_models)):
    xtest.append(input_models[x - timetopredict:x, 0])

xtest = np.array(xtest)
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

# Predict stock prices using the trained model
testprice = model.predict(xtest)

# Inverse transform the predicted prices to the original scale
testprice = scaler.inverse_transform(testprice)

# Plot the actual test prices and predicted prices
plt.plot(actual_test_price, label="real")
plt.plot(testprice, label="predict")
plt.legend()
plt.show()

# Get the last date in the dataset
last_date = data["Date"].max()

# Generate future dates for the next 90 days
future_dates = [last_date + timedelta(days=i) for i in range(1, 91)]

# Prepare input data for predicting future stock prices
input_future = total_dataset[-timetopredict:].values
input_future = input_future.reshape((-1, 1))
input_future = scaler.transform(input_future)

x_future = []

# Create input data for the next 90 days
for i in range(timetopredict, timetopredict + 90):
    x_future.append(input_future[i - timetopredict:i, 0])

# Ensure that all arrays in x_future have the same length
max_len = max(len(arr) for arr in x_future)
x_future = [np.pad(arr, (0, max_len - len(arr))) for arr in x_future]

# Concatenate the arrays to create the input for LSTM
x_future = np.concatenate(x_future, axis=0)

# Reshape for LSTM input (samples, time steps, features)
x_future = np.reshape(x_future, (int(x_future.shape[0] / timetopredict), timetopredict, 1))

# Predict future stock prices using the trained model
future_predictions = model.predict(x_future)

# Inverse transform the predictions to the original scale
future_predictions = scaler.inverse_transform(future_predictions)

# Flatten the predictions
future_predictions = future_predictions.flatten()

# Plot historical data along with predicted prices for the next 90 days
plt.plot(data["Date"], data["Close"], label="Historical Data")
plt.plot(future_dates, future_predictions, label="Future Predictions")

plt.title('Historical and Future Stock Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
