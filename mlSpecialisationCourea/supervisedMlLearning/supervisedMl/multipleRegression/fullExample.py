import numpy as np
import matplotlib.pyplot as plt
import time

from lab_utils_uni import (
    plt_intuition,
    plt_stationary,
    plt_update_onclick,
    soup_bowl
)

plt.style.use('./deeplearning.mplstyle')


x_singleTrain = np.array(
    [1.0, 1.7, 2.0, 2.5, 3.0, 3.2],
    )

y_singleTrain = np.array(
    [250, 300, 480, 430, 630, 730,],
    )

x_train = np.array(
    [1.0, 1.7, 2.0, 2.5, 3.0, 3.2],
    [2.0, 2.7, 3.0, 4.5, 4.0, 4.2],
    [3.0, 3.7, 4.0, 3.5, 4.0, 5.2],
    )
y_train = np.array(
    [250, 300, 480, 430, 630, 730,],
    [350, 400, 580, 530, 730, 830,],
    [450, 500, 680, 630, 830, 930,],
    )

def compute_cost(x, y, w, b):
    m = x.shape[0]
    Y = w * x + b
    return (1 / (2 * m)) * ((Y - y) ** 2).sum()

def singleRegression (x, y):
    m = x.shape[0]
    xMean = x.mean()
    yMean = y.mean()

    w = ((x - xMean) * (y - yMean)).sum() / ((x - xMean)**2).sum()
    b = yMean - w * xMean
    Y = w * x + b

    jwb = 1 / (2 * m) * ((Y - y)**2).sum()

    return jwb, w, b, Y

def gradientDescent (x, y, w, b, m):
    x.shape[0]
    Y = w * x + b 

    dW = (1 / m) * ((Y - y) * x).sum()
    dB = (1 / m) * ((Y - y)).sum()

    return dW, dB

def loop(x, y, w, b, a, iterations):
    m = x.shape[0]
    jHistory = []
    wbHistory = []

    for i in range(iterations):
        dW, dB = gradientDescent (x, y, w, b, m)

        w = w - a * dW
        b = b - a * dB

        if i < 1000:
            jHistory.append(compute_cost(x, y, w, b))
            wbHistory.append((w, b))
        
    return w, b, jHistory, wbHistory

w = 0
b = 0 
w, b, jHist, wbHist = loop(x_singleTrain, y_singleTrain, w, b, 0.01, 1000)










