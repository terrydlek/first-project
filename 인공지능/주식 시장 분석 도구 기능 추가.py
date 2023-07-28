import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Matplotlib 백엔드를 설정 (Tkinter 대신 다른 백엔드 사용)
matplotlib.use('TkAgg')  # Agg 백엔드 사용 (렌더링하지 않고 그림 파일로 저장)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def calculate_technical_indicators(data):
    data['SMA'] = data['Close'].rolling(window=20).mean()
    data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    return data

def calculate_rsi(close_prices, window=14):
    delta = close_prices.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def reshape_data_for_decision_tree(X):
    return X.reshape(X.shape[0], -1)

def preprocess_data(X):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed

def train_decision_tree_model(X_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

def train_neural_network_model(X_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


if __name__ == '__main__':
    symbol = 'AAPL'  # 분석할 주식 종목 심볼 (Apple Inc.의 경우)
    start_date = '1980-01-01'  # 시작일
    end_date = '2023-06-01'  # 종료일

    # 주식 데이터 가져오기
    stock_data = get_stock_data(symbol, start_date, end_date)

    # 기술적 지표 계산
    stock_data = calculate_technical_indicators(stock_data)

    # 주가 예측을 위한 데이터셋 생성
    X, y = create_dataset(stock_data[['SMA', 'EMA', 'MACD', 'RSI']].values, look_back=5)

    # 데이터셋 분할 (Train, Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # NaN 값 처리
    X_train_imputed = preprocess_data(reshape_data_for_decision_tree(X_train))
    X_test_imputed = preprocess_data(X_test.reshape(X_test.shape[0], -1))

    # NaN 값이 있는 행 제거
    nan_indices_train = np.where(~np.isnan(y_train))[0]
    y_train = y_train[~np.isnan(y_train)]
    X_train_imputed = X_train_imputed[nan_indices_train]

    nan_indices_test = np.where(~np.isnan(y_test))[0]
    y_test = y_test[~np.isnan(y_test)]
    X_test_imputed = X_test_imputed[nan_indices_test]

    # 의사결정트리 회귀 모델 학습 및 평가
    dt_regressor = DecisionTreeRegressor()
    dt_regressor.fit(X_train_imputed, y_train)
    dt_y_pred = dt_regressor.predict(X_test_imputed)
    dt_r2_score = r2_score(y_test, dt_y_pred)
    dt_mse = mean_squared_error(y_test, dt_y_pred)

    # 신경망 회귀 모델 학습 및 평가
    nn_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=200)
    nn_regressor.fit(X_train_imputed, y_train)
    nn_y_pred = nn_regressor.predict(X_test_imputed)
    nn_r2_score = r2_score(y_test, nn_y_pred)
    nn_mse = mean_squared_error(y_test, nn_y_pred)

    print(f'의사결정트리 회귀 모델 R^2 스코어: {dt_r2_score:.2f}')
    print(f'의사결정트리 회귀 모델 MSE: {dt_mse:.2f}')
    print(f'신경망 회귀 모델 R^2 스코어: {nn_r2_score:.2f}')
    print(f'신경망 회귀 모델 MSE: {nn_mse:.2f}')

    # 시뮬레이션 결과 시각화
    stock_data['Signal'] = 0
    stock_data.loc[stock_data['MACD'] > 0, 'Signal'] = 1  # 매수 신호
    stock_data.loc[stock_data['MACD'] < 0, 'Signal'] = -1  # 매도 신호

    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label='stock price', color='blue')
    plt.plot(stock_data[stock_data['Signal'] == 1].index, stock_data['Close'][stock_data['Signal'] == 1], '^', markersize=10, color='g', label='Buy Signal')
    plt.plot(stock_data[stock_data['Signal'] == -1].index, stock_data['Close'][stock_data['Signal'] == -1], 'v', markersize=10, color='r', label='Sell Signal')
    plt.title(f'{symbol} buy/sell simulation')
    plt.xlabel('date')
    plt.ylabel('stock price')
    plt.legend()
    plt.grid(True)
    plt.show()
