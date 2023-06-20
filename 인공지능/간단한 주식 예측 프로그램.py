# 주식 데이터를 수집하고, 데이터를 기반으로 한 간단한 예측 모델
import yfinance as yf

# 주식 데이터 가져오기
stock_data = yf.download('AAPL', start='2018-01-01', end='2023-06-01')
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 데이터 전처리
stock_data['Date'] = stock_data.index
stock_data.reset_index(drop=True, inplace=True)

# 특성과 타겟 변수 선택
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

X = stock_data[features]
y = stock_data[target]

# 학습 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 데이터로 예측 수행
y_pred = model.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

import matplotlib.pyplot as plt

plt.plot(y_test.index, y_test, label='actual value')
plt.plot(y_test.index, y_pred, label='predicted value')
plt.xlabel('date')
plt.ylabel('the closing price of a stock')
plt.title('Stock closing forecast results')
plt.legend()
plt.show()
