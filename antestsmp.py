import numpy as np
import pandas as pd
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt


model = load_model("C:\\Users\\vikas\\Stock Predictions Model.keras")

import pandas as pd
df = pd.read_csv("C:\\Users\\vikas\\OneDrive\\Desktop\\College\\Stock Market Prediction\\NSE_1001_TO_1050_start_to_15082020.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)

# Identify unique stock symbols
unique_symbols = df["SYMBOL"].unique()
unique_symbols_list = list(unique_symbols)

# Loop through the list and print each element with its index
for index, stock in enumerate(unique_symbols_list):
    print(f"Stock at {index}: {stock}")

segregated_data = {symbol: df[df["SYMBOL"] == symbol] for symbol in unique_symbols}
n = int(input("Enter Stock Number Which you want to Analyze:"))


if n == 0:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])

if n == 1:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==2:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n==3:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])

elif n==4:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n==5:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n==6:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n==7:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])

elif n ==8:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])

elif n ==9:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])

elif n ==10:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])

elif n ==11:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==12:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==13:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==14:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==15:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==16:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==17:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==18:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==19:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==20:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==21:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==22:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==23:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==24:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==25:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==26:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==27:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==28:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==29:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==30:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==31:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==32:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==33:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==34:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==35:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==36:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==37:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==38:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==39:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==40:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==41:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==42:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==43:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==44:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==45:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==46:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==47:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==48:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])
    
elif n ==49:
    example_symbol = unique_symbols[n]
    print(f"Data for {example_symbol}:")
    print(segregated_data[example_symbol])

print(segregated_data[example_symbol].drop('SYMBOL', axis=1))

st.header('Stock Market Predictor')
st.subheader('Stock Data')
st.write(segregated_data[example_symbol])

data_train = pd.DataFrame(segregated_data[example_symbol].Close[0: int(len(segregated_data[example_symbol])*0.80)])
data_test = pd.DataFrame(segregated_data[example_symbol].Close[int(len(segregated_data[example_symbol])*0.80): len(segregated_data[example_symbol])])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)


st.subheader('Price vs MA50')
ma_50_days = segregated_data[example_symbol].Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(segregated_data[example_symbol].Close, 'g')
plt.show()
st.pyplot(fig1)


st.subheader('Price vs MA50 vs MA100')
ma_100_days = segregated_data[example_symbol].Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(segregated_data[example_symbol].Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = segregated_data[example_symbol].Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(segregated_data[example_symbol].Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale


st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)