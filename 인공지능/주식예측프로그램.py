'''
삼성전자 주가 데이터는 야후 파이낸스에서 csv 파일로 다운로드 할 수 있음.
csv 파일에는 7개의 column만이 존재하나, 예측의 정확도를 높이기 위해 3일 이동평균선(3MA)', 5일 평균선(5MA) 데이터 추가함

총 4단계
1. 데이터 로드 및 분호 확인
df = pd.read_csv(), df.describe(), df.hist(), plot() 등
↓
2. 데이터 전처리
1) outlier / missing value 확인 후 대체(또는 삭제) 처리
2) 데이터 정규화 / 표준화
3) 딥러닝 학습을 위한 feature column / label column 정의
↓
3. 데이터 생성
1) window size(몇 개의 데이터를 이용해 정답을 나타낼 것인지) 설정 후 feature(입력 데이터) / label(정답) 시계열 데이터 생성
2) 학습 데이터 생성, 이 때 입력 데이터는 (batch size(총 몇개의 window size가 있는지), time steps(몇개의 데이터를 이용해 정답을 나타내는지?), input dims(한번에 들어가는 데이터의 갯수가 몇개인지)) 형태의 3차원 텐서로 생성되어야 함
↓
4. 순환신경망 모델 구축 및 학습

'''

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


'''1. 데이터 로드 및 분포 확인'''
raw_df = pd.read_csv('./005930.KS_3MA_5MA.csv') # 데이터파일을 불러옴
raw_df.head() # 불러온 데이터 중 adj close 값을 예측할 계획

# 삼성전자의 주가는 꾸준히 우상향하는 그래프를 띔
# LSTM을 이용해 20년간의 주가 데이터를 학습
plt.figure(figsize=(7, 4))
plt.title('SAMSUNG ELECTRONIC STOCK PRICE')
plt.ylabel("price (won)")
plt.xlabel('period (day)')
plt.grid()

plt.plot(raw_df['Adj Close'], label = 'Adj Close', color='b')
plt.legend(loc='best')
plt.show()


'''2. 데이터 전처리 Outlier 확인'''
# 통계적으로 비정상적으로 크거나 작은 데이터인 outlier(특이값)는 딥러닝 학습을 하기 위해서는 적절한 값으로 바꾸거나 삭제하는 등의 처리가 반드시 필요!
# 판다스 describe()를 통해서 삼전 주가 데이터 통계를 확인해보면, 거래량을 나타내는 volume 최소값이 0임을 알 수 있음
# => 주식과 같은 금융데이터에서 volume(거래량) 값이 없는, 즉 0으로 나타나는 곳은 missing value(결측값)인 NaN으로 취급하는 것이 일반적
raw_df.describe()

# 결측치는 특정 데이터가 누락된 것을 말함, outlier와 마찬가지로 이러한 missing value를 제가ㅓ하거나 적절한 값으로 대체하는 등의 처리가 필요함
# 판다스 isnull().sum() 을 통해 삼전 주가 데이터의 missing value를 확인해보면, 6개의 칼럼에서 각각 6개 missing value가 있음을 알 수 있음
# => 주식과 같은 금융데이터에서 NaN으로 표시되는 missing value는 평균값이나 중간값 등으로 대체하지 않고 해당되는 행 전체를 삭제하는 것이 일반적임
raw_df.isnull().sum()
raw_df.loc[raw_df['Open'].isna()]

# volume 값 0을 모두 NaN으로 대체함
# replace를 활용해 0을 non 으로 만든 후 pandas가 이러한 outlier도 missing value로 인식하게 만듦
raw_df['Volume'] = raw_df['Volume'].replace(0, np.nan)

# 각 column에 0의 개수를 확인
for col in raw_df.columns:
  missing_rows = raw_df.loc[raw_df[col] == 0].shape[0]
  print(col + ':' + str(missing_rows))

# 모든 missing value 삭제
# 이제 우리가 분석할 데이터에는 outlier나 missing value가 모두 없어졌다는 것을 알 수 있음
raw_df = raw_df.dropna()
raw_df.isnull().sum()


'''2. 데이터 전처리 - 정규화 '''
# 딥러닝 학습이 잘 되기 위해서는 정규화 작업이 필요함. 즉 날짜를 나타내는 date 항목을 제외한 숫자로 표현되는 모든 column에 대해 0 ~ 1 값으로 정규화 수행
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# 정규화 대상 column 정의
scale_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', '3MA', '5MA', 'Volume']
# fit transform 함수를 사용해 0과 1사이의 값을 갖도록 정규화를 수행
scaled_df = scaler.fit_transform(raw_df[scale_cols])
# 리턴 값은 넘파이
print(type(scaled_df), '\n')
# 정규화된 새로운 DataFrame 생성
scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)
print(scaled_df)


'''2. 데이터 전처리 - feature column / label column'''
# 딥러닝 학습을 위한 입력데이터 feature column, 정답데이터 label column 정의 후 numpy로 변환하여 데이터 전처리 과정을 완료
# feature 정의(입력데이터)
feature_cols = ['3MA', '5MA', 'Adj Close']
label_cols = ['Adj Close']
label_df = pd.DataFrame(scaled_df, columns=label_cols)
feature_df = pd.DataFrame(scaled_df, columns=feature_cols)
print(feature_df)
print(label_df)
# 딥러닝 학습을 위해 dataframe을 numpy로 변환
label_np = label_df.to_numpy()
feature_np = feature_df.to_numpy()
# 데이터 전처리 과정 끝


'''3. 데이터 생성 - 입력데이터 feature / 정답데이터 label'''
# 1) 넘파이로 주어지는 시계열 데이터 feature_np, label_np로부터, window size에 맞게 RNN 입력 데이터 X, 정답데이터 Y 생성함
# 이 때 리턴되는 입력 데이터 X.shape() = (batch size, time steps, input dims)
# 2) feature[i:i + window_size] 슬라이싱 이용해 [[..],[..],..] 형상으로 입력 데이터 즉 feature를 생성함
# 3) feature_list = [[..],[..],..] 이므로 리턴 값 np.array(feature_list)는 (batch size, time steps, input dims)형상 가짐

# 입력 파라미터 feature, label => numpy type
def make_sequence_dataset(feature, label, window_size):
    feature_list = []  # 생성될 feature list
    label_list = []  # 생성될 label list

    for i in range(len(feature) - window_size):
        feature_list.append(feature[i:i + window_size])
        label_list.append(label[i + window_size])

    return np.array(feature_list), np.array(label_list)

# 1) 학습 데이터 X, Y 생성
window_size = 40
X, Y = make_sequence_dataset(feature_np, label_np, window_size)
print(X.shape, Y.shape)

# 2) 트레이닝 데이터 / 테스트 데이터 분리
split = -200
x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# 3) LSTM 모델 구축
model = Sequential()
# LSTM 계층에 tanh를 활성화 함수로 가지는 노드 수 128개
model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape))
# 128개의 노드를 가지는 lstm 레이어와 한개의 레이어를 가지는 출력층으로 구성된 모델
model.add(Dense(1, activation="linear"))
model.summary()



# 4) 모델 컴파일 및 학습
from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 100, batch_size = 16, callbacks=[early_stop])
# 손실함수 값이 지속적으로 작아지다가 더 이상 줄어들 지 않을 때 조기 종료 됨


# 5) 삼성전자 주가 예측
pred = model.predict(x_test)
plt.figure(figsize=(12, 6))
plt.title('3MA + 5MA + Adj Close, window_size = 40')
plt.ylabel('adj close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(pred, label = 'prediction')
plt.grid()
plt.legend(loc='best')
plt.show()


# 고찰: 예측한 prediction 값이 차이가 크지 않은 것을 확인할 수 있었음
# 좀 더 높은 정확도를 통해 수익을 얻고 싶다면 어떤 feature가 수익에 영향을 주는지 , feature의 상관관계 등을 파악하는 등의 domain knowledge가 필요할 듯