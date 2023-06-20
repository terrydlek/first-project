import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 주식 데이터 가져오기
stock_data = yf.download('AAPL', start='2018-01-01', end='2023-06-01')

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

# 데이터셋 분할 (학습:검증 = 8:2)
split_index = int(len(X) * 0.8)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# 테스트 데이터 예측
scaled_test_data = scaler.transform(data[len(data)-len(X):])
X_test = []
y_test = []
for i in range(len(scaled_test_data) - 60, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-60:i, 0])
    y_test.append(scaled_test_data[i, 0])
X_test = np.array(X_test)
y_test = np.array(y_test)

predictions = model.predict(X_test)

# 예측 결과 확인
y_pred = scaler.inverse_transform(predictions)

# 실제값과 예측값 비교
start_index = split_index + 60  # 실제값의 시작 인덱스
end_index = start_index + len(y_test)  # 실제값의 끝 인덱스
df = pd.DataFrame({'실제값': data[start_index:end_index, 0], '예측값': y_pred.flatten()})
print(df)



