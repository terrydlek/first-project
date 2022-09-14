import sys
import os
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from mnist import load_mnist
from functions import sigmoid, softmax

x, t = get_data()