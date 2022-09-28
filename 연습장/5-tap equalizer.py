import numpy as np
Pr = np.array([[1/5, 1/10, 1/17, 1/26, 1/37],
      [1, 1/2, 1/5, 1/10, 1/17],
      [1/5, 1/2, 1, 1/2, 1/5],
      [1/7, 1/10, 1/5, 1/2, 1],
      [1/37, 1/26, 1/17, 1/10, 1/5]])

# Pr의 역행렬 구하기
Pr_inverse = np.linalg.inv(Pr)
Po = np.array([[0], [0], [1], [0], [0]])

C = np.dot(Pr_inverse, Po)
print(C)
