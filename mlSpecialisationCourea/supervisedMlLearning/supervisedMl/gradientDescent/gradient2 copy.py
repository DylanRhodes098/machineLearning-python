import math, copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import (
    plt_house_x, 
    plt_contour_wgrad, 
    plt_divergence, 
    plt_gradients
)

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430, 630, 730])
m = x_train.shape[0] 
a = 0.01
iterations = 1000

###CostFunction###

#Short fwb function#
def fwb (x, y):
    xMean = x.mean()
    yMean = y.mean()

    numerator = ((x - xMean) * (y - yMean)).sum()
    denominator = ((x - xMean) ** 2).sum()

    w = numerator / denominator
    b = yMean - w * xMean
    fwb = w * x + b

    return fwb, w, b

fwb, w, b = fwb(x_train, y_train)

#Shortened Function#
def compute_cost(x, y, w, b): 

    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum 

    return total_cost   

###Gradient Descent###

def fwb(x, w, b):
    return w * x + b

def computeDJ(x, y, fwb_vals):
    m = x.shape[0]
    dW = (1/m) * ((fwb_vals - y) * x).sum()
    dB = (1/m) * (fwb_vals - y).sum()
    return dW, dB

def computeGradient(w, b, dW, dB, a):
    newW = w - a * dW
    newB = b - a * dB
    return newW, newB

def loop(x, y, w_init, b_init, a, numiters):
    w = w_init
    b = b_init

    jHistory = []
    wbHistory = []

    for i in range(numiters):
        # 1) predictions
        preds = fwb(x, w, b)

        # 2) cost (optional, for history/plotting)
        cost = ((preds - y) ** 2).sum() / (2 * x.shape[0])

        # 3) gradients
        dW, dB = computeDJ(x, y, preds)

        # 4) update w, b
        w, b = computeGradient(w, b, dW, dB, a)

        # 5) store history (e.g. first 1000 points)
        if i < 1000:
            jHistory.append(cost)
            wbHistory.append((w, b))

    return w, b, jHistory, wbHistory