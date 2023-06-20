import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 주식 데이터 가져오기
stock_data = yf.download('AAPL', start='2018-01-01', end='2023-06-20')

# 종가 데이터 추출
data = stock_data['Close'].values.reshape(-1, 1)

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 데이터셋 생성
X = []
y = []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])
X = np.array(X)
y = np.array(y)

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X, y, epochs=10, batch_size=32)

# 다음날 데이터 예측을 위한 데이터 준비
last_60_days = scaled_data[-60:]
X_next_day = np.array([last_60_days])
X_next_day = np.reshape(X_next_day, (X_next_day.shape[0], X_next_day.shape[1], 1))

# 다음날 종가 예측
predicted_price = model.predict(X_next_day)

# 예측값 역정규화
predicted_price = scaler.inverse_transform(predicted_price)

# 다음날 종가 예측값 출력
print("다음날 종가 예측값:", predicted_price)
