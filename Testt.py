import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data for candlestick chart
df = pd.read_csv("TSLA.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Plot candlestick chart using plotly
fig_candlestick = go.Figure(data=[go.Candlestick(x=df["Date"],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'])])
fig_candlestick.show()

# Preprocess data for LSTM model
data = df.dropna()

# Exclude the datetime column from scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

timetopredict = 90

xtrain = []
ytrain = []

# Create input sequences and corresponding output
for x in range(timetopredict, len(scaled_data)):
    xtrain.append(scaled_data[x - timetopredict:x, 0])
    ytrain.append(scaled_data[x, 0])

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

# Build and train the LSTM model
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

# Prepare test data
test_start_date = data["Date"].iloc[-timetopredict:]  # Adjusted index
test_data = data[data["Date"] >= test_start_date.iloc[0]]
test_data = pd.merge(data["Close"], test_data)

actual_test_price = test_data["Close"].values

# Exclude the datetime column from the input data
input_models = scaler.transform(test_data["Close"].values.reshape(-1, 1))

xtest = []

# Create input sequences for testing
for x in range(timetopredict, len(input_models)):
    xtest.append(input_models[x - timetopredict:x, 0])

xtest = np.array(xtest)
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

# Predict future stock prices using the trained model
predicted_prices = model.predict(xtest)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot actual vs predicted prices
plt.plot(data["Date"], data["Close"], label="Actual Prices")
plt.plot(test_data["Date"], actual_test_price, label="Test Data")
plt.plot(test_data["Date"].iloc[:len(predicted_prices)], predicted_prices,
         label="Predicted Prices", linestyle="--", color="orange")
plt.legend()
plt.show()
