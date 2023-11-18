import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


# 주식 데이터 가져오기
stock_data = yf.download('AAPL', start='2020-01-01', end='2023-11-01')

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
model.fit(X, y, epochs=100, batch_size=32)

# 다음날 데이터 예측을 위한 데이터 준비
last_60_days = scaled_data[-60:]
X_next_day = np.array([last_60_days])
X_next_day = np.reshape(X_next_day, (X_next_day.shape[0], X_next_day.shape[1], 1))

# 실제 데이터
plt.plot(stock_data.index[-len(X_next_day[0]):], stock_data['Close'].iloc[-len(X_next_day[0]):], label='실제 종가', color='blue')

# 다음날 종가 예측
predicted_price = model.predict(X_next_day)

# 예측값 역정규화
predicted_price = scaler.inverse_transform(predicted_price)

# 다음날 종가 예측값을 시각화
plt.figure(figsize=(10, 6))

# 예측값
plt.plot(stock_data.index[-1] + pd.DateOffset(1), predicted_price[0][0], marker='o', markersize=8, label='다음날 예측 종가', color='red')

plt.title('다음날 종가 예측')
plt.xlabel('날짜')
plt.ylabel('종가')
plt.legend()
plt.show()

# 주식 종가 예측값과 실제 종가의 정확도 계산
accuracy = 1 - abs((predicted_price[0][0] - stock_data['Close'].iloc[-1]) / stock_data['Close'].iloc[-1])

# 정확도 출력
print("주식 종가 예측 정확도:", round(accuracy * 100, 2), "%")


# 다음날 종가 예측값 출력
print("다음날 종가 예측값:", predicted_price)
