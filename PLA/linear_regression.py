"""
Created on  2019-10-15
@author: Jingchao Yang
"""
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv

# generating dataset
X, y, coefficients = make_regression(
    n_samples=50,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=5,
    coef=True,
    random_state=1
)
y = y.reshape(len(y), 1)

# pseudo-inverse
pseudo_inverseX = np.dot(inv(np.dot(np.transpose(X), X)), np.transpose(X))

# getting weights
w = np.dot(pseudo_inverseX, y)

# predicted
pre = w * X

plt.scatter(X, y)
plt.plot(X, pre, c='red')
plt.show()
