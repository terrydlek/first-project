import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data


def plot_stock_price(data, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'])
    plt.title(f'{symbol} stock')
    plt.xlabel('date')
    plt.ylabel('stock price')
    plt.grid(True)
    plt.show()


def calculate_returns(data):
    data['Returns'] = data['Close'].pct_change()
    return data


def plot_returns(data, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Returns'])
    plt.title(f'{symbol} stock earning rate')
    plt.xlabel('date')
    plt.ylabel('stock earning rate')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    symbol = 'AAPL'  # 분석할 주식 종목 심볼 (Apple Inc.의 경우)
    start_date = '2018-01-01'  # 시작일
    end_date = '2023-01-01'  # 종료일

    # 주식 데이터 가져오기
    stock_data = get_stock_data(symbol, start_date, end_date)

    # 주식 가격 그래프 그리기
    plot_stock_price(stock_data, symbol)

    # 주식 수익률 계산 및 그래프 그리기
    stock_data = calculate_returns(stock_data)
    plot_returns(stock_data, symbol)
