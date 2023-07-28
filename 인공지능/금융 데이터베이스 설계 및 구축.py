import yfinance as yf
import sqlite3
from datetime import datetime, timedelta
import time
import pandas as pd
import matplotlib.pyplot as plt

# 데이터베이스 연결
conn = sqlite3.connect('financial_database.db')
cursor = conn.cursor()

# 로깅 기능 추가
import logging
logging.basicConfig(filename='financial_data.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 주식 테이블 생성
def create_stock_table():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL
        )
    ''')
    conn.commit()

# 주식 데이터 삽입
def insert_stock_data(symbol, date, open_price, high_price, low_price, close_price, volume):
    cursor.execute('''
        INSERT INTO stocks (symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, date, open_price, high_price, low_price, close_price, volume))
    conn.commit()

# 데이터 수집 및 데이터베이스에 추가
def collect_and_insert_stock_data(symbol):
    # 어제 날짜와 오늘 날짜 계산
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    # 주식 데이터 가져오기
    try:
        stock_data = yf.download(symbol, start=yesterday, end=today)
    except Exception as e:
        logging.error(f"Error while collecting data for {symbol}: {str(e)}")
        return

    # 데이터베이스에 주식 데이터 삽입
    for index, row in stock_data.iterrows():
        insert_stock_data(symbol, index.strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Volume'])
    logging.info(f"Data collected and inserted for {symbol}")

# 기술적 지표 계산
def calculate_technical_indicators(symbol):
    cursor.execute(f"SELECT * FROM stocks WHERE symbol = ? ORDER BY date ASC", (symbol,))
    stock_data = cursor.fetchall()

    df = pd.DataFrame(stock_data, columns=['id', 'symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
    df.set_index('date', inplace=True)

    df['SMA'] = df['close'].rolling(window=20).mean()
    df['EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['close'])

    return df

def calculate_rsi(close_prices, window=14):
    delta = close_prices.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 데이터 시각화
def visualize_data(symbol):
    df = calculate_technical_indicators(symbol)
    plt.figure(figsize=(12, 6))
    plt.plot(df['close'], label='stock price', color='blue')
    plt.plot(df[df['MACD'] > 0].index, df['close'][df['MACD'] > 0], '^', markersize=10, color='g', label='Buy Signal')
    plt.plot(df[df['MACD'] < 0].index, df['close'][df['MACD'] < 0], 'v', markersize=10, color='r', label='Sell Signal')
    plt.title(f'{symbol} buy/sell simulation')
    plt.xlabel('date')
    plt.ylabel('stock price')
    plt.legend()
    plt.grid(True)
    plt.show()

# 주식 데이터베이스 테이블 생성
create_stock_table()

# 기본적으로 AAPL, GOOG, AMZN 주식 데이터를 DB에 추가
symbols = ['AAPL', 'GOOG', 'AMZN']
for symbol in symbols:
    collect_and_insert_stock_data(symbol)

# 매일 정해진 시간에 데이터 수집 및 데이터베이스에 추가
while True:
    now = datetime.now()
    if now.hour == 19 and now.minute == 44:  # 매일 오후 6시에 실행
        for symbol in symbols:
            collect_and_insert_stock_data(symbol)
    keyboard = input()
    if keyboard == "q":
        break
    else:
        pass
    time.sleep(5)  # 1분마다 체크

# 데이터 시각화
print("Enter a stock symbol to visualize data or 'EXIT' to quit:")
while True:
    user_input = input().upper()
    if user_input == 'EXIT':
        break
    if len(user_input) > 0:
        visualize_data(user_input)

# 데이터베이스 연결 종료
conn.close()
