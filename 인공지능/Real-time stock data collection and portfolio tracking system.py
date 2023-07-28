import yfinance as yf
import sqlite3
from datetime import datetime, timedelta
import time
import pandas as pd
import matplotlib.pyplot as plt
import threading

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

# 주식 데이터 삽입 또는 업데이트
def insert_or_replace_stock_data(symbol, date, open_price, high_price, low_price, close_price, volume):
    cursor.execute('''
        INSERT OR REPLACE INTO stocks (symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, date, open_price, high_price, low_price, close_price, volume))
    conn.commit()

# 실시간 주식 데이터 가져오기
def get_realtime_stock_price(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d")
    if not data.empty:
        row = data.iloc[-1]
        return row['Close']
    return None

# 데이터 수집 및 데이터베이스에 추가 또는 업데이트
def collect_and_upsert_stock_data(symbol):
    # 어제 날짜와 오늘 날짜 계산
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    # 주식 데이터 가져오기
    try:
        stock_data = yf.download(symbol, start=yesterday, end=today)
    except Exception as e:
        logging.error(f"Error while collecting data for {symbol}: {str(e)}")
        return

    # 데이터베이스에 주식 데이터 추가 또는 업데이트
    for index, row in stock_data.iterrows():
        insert_or_replace_stock_data(symbol, index.strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Volume'])
    logging.info(f"Data collected and upserted for {symbol}")

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

# 포트폴리오 추적 기능
def track_portfolio(portfolio):
    total_investment = 0
    current_value = 0
    for symbol, quantity in portfolio.items():
        cursor.execute(f"SELECT * FROM stocks WHERE symbol = ? ORDER BY date DESC LIMIT 1", (symbol,))
        stock_data = cursor.fetchone()
        if stock_data:
            stock_price = stock_data[6]  # 'close' column
            total_investment += stock_price * quantity
            current_value += get_realtime_stock_price(symbol) * quantity

    if total_investment == 0:
        return "Portfolio is empty."
    else:
        profit_loss = current_value - total_investment
        return f"Total Investment: {total_investment}, Current Value: {current_value}, Profit/Loss: {profit_loss}"

# 주식 데이터베이스 테이블 생성
create_stock_table()

# 기본적으로 AAPL, GOOG, AMZN 주식 데이터를 DB에 추가
symbols = ['AAPL', 'GOOG', 'AMZN']
for symbol in symbols:
    collect_and_upsert_stock_data(symbol)

# 키보드 입력 쓰레드
def keyboard_input_thread():
    global is_input_received
    is_input_received = False
    while True:
        keyboard = input()
        if keyboard == "q":
            break
        is_input_received = True

# 키보드 입력 쓰레드 시작
keyboard_thread = threading.Thread(target=keyboard_input_thread)
keyboard_thread.start()

# 메인 루프 시작
while True:
    now = datetime.now()
    if now.hour == 19:  # 매일 오후 6시에 실행
        for symbol in symbols:
            collect_and_upsert_stock_data(symbol)
        logging.info("Data update completed")

    # 5초마다 키보드 입력 체크
    for _ in range(5):
        if is_input_received:
            break
        time.sleep(1)
    is_input_received = False

    if keyboard_thread.is_alive():
        continue
    else:
        break

# 포트폴리오 추적
portfolio = {'AAPL': 10, 'GOOG': 5, 'AMZN': 3}  # 사용자의 보유 주식 포트폴리오 (예시)
result = track_portfolio(portfolio)
print(result)

# 키보드 입력 쓰레드 종료 대기
keyboard_thread.join()

# 데이터베이스 연결 종료
conn.close()
