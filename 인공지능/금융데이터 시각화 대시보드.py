import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 대시보드에서 모니터링할 주식 심볼
symbol = 'BTC-USD'

# 주식 데이터 가져오기 함수
def get_stock_data(symbol):
    data = yf.download(symbol, start='2023-07-25', end='2023-07-30')
    return data

# 주식 데이터를 실시간으로 모니터링하는 함수
def update_dashboard(i):
    stock_data = get_stock_data(symbol)

    # 주가 그래프 그리기
    plt.clf()
    plt.plot(stock_data['Close'], label='stock price', color='blue')
    plt.title(f'{symbol} stock price monitering')
    plt.xlabel('date')
    plt.ylabel('stock price')
    plt.legend()
    plt.grid(True)

# 대시보드 업데이트 간격 (밀리초 단위)
update_interval = 5000  # 5초

# 대시보드 그래프 업데이트
ani = FuncAnimation(plt.gcf(), update_dashboard, interval=update_interval)

# 대시보드 실행
plt.show()
