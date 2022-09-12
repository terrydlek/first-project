import numpy as np
import matplotlib.pylab as plt
#-------------------------------------------
#평균제곱 오차
#정답은 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#예1 : '2'일 확률이 가장 높다고 추정(0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

def mean_squared_error(y, t): #평균 제곱 오차
    return 0.5 * np.sum((y - t)**2)

print(mean_squared_error(np.array(y), np.array(t)))
#예2 : '7'일 확률이 가장 높다고 추정(0.6)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))
 #==> 첫 번째 예의 손실 함수 쪽 출력이 작으며 정답 레이블과의 오차도 작은 것을 알 수 있음.
 #즉 평균 제곱 오차를 기준으로는 첫 번째 추정 결과가 오차가 더 작으므로 정답에 더 가까울 것으로 판단할 수 있다.
#-------------------------------------------

#교차 엔트로피 오차
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y + delta))


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
#==> 첫 번째 예는 정답일 때의 출력이 0.6인 경우로, 이때의 교차 엔트로피 오차는 약 0.51임.
# 그 다음은 정답일 때의 출력이 더 낮은 0.1인 경우로, 이때의 교차 엔트로피 오차는 무려 2.3임
#즉, 결과(오차 값)가 더 작은 첫 번째 추정이 정답일 가능성이 높다고 판단한 것으로, 앞서 평균 제곱 오차의 판단과 일치
#-------------------------------------------
#미니배치 학습
import sys, os
sys.path.append(os.pardir)
import numpy as np
import mnist
(x_train, t_train), (x_test, t_test) = mnist.load_mnist(normalize=True, one_hot_label=True)
#load_mnist는 MNIST 데이터셋을 읽어오는 함수이다.
print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
#-------------------------------------------
#(배치용) 교차 엔트로피 오차 구현하기


def cross_entropy_error1(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np/sum(t*np.log(y)) / batch_size


#정답 레이블이 원-핫 인코딩이 아니라 2나 7 등의 숫자 레이블로 주어졌을 때의 교차 엔트로피 오차 구현하기
def cross_entropy_error2(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
#-------------------------------------------


#미분함수 구현
def numerical_diff(f, x):
    h = 1e-4 #0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


#아주 작은 차분으로 미분하는 것을 수치 미분이라고 함.
def function_1(x):
    return 0.01*x**2 + 0.1*x


'''x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
'''
print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))
