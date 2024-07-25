import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import streamlit as st
import keras
import matplotlib.pyplot as plt
 

model=keras.models.load_model("stockModel.h5")

st.header('Stock Market Predictor')

stock=st.text_input('Enter Stock Symbol','GOOG')
start='2012-01-01'
end='2022-12-31'

data=yf.download(stock,start,end)

st.subheader('Stock Data')
st.write(data)
data_train=pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

MA_50=data.Close.rolling(50).mean()
MA_100=data.Close.rolling(100).mean()
MA_200=data.Close.rolling(200).mean()
st.subheader('MA 50')
fig1=plt.figure(figsize=(8,6))
plt.plot(MA_50,'r',label='Moving Average 50')
plt.plot(data.Close,'b',label='Original Price')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader('MA_50 vs MA 100 vs Price')
fig2=plt.figure(figsize=(8,6))
plt.plot(MA_50,'g',label='MA 50')
plt.plot(MA_100,'r',label='MA 100')
plt.plot(data.Close,'b',label='Original')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader('MA_100 vs MA_200 vs Price')
fig3=plt.figure(figsize=(8,6))
plt.plot(MA_100,'g',label='MA 100')
plt.plot(MA_200,'r',label='MA 200')
plt.plot(data.Close,'b',label='Original Price')
plt.legend()
plt.show()
st.pyplot(fig3)

past_100days=data_train.tail(100)

data_test=pd.concat([past_100days,data_test],ignore_index=True)
data_test_scale=scaler.fit_transform(data_test)

x=[]
y=[]

for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y=np.array(x),np.array(y)

predict=model.predict(x)

scale=1/scaler.scale_

predict=predict*scale
y=y*scale

st.subheader('Original Price vs Predicted Price')
fig4=plt.figure(figsize=(8,6))
plt.plot(predict,'r',label='Predicted Price')
plt.plot(y,'g',label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)