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

#Means#
x_mean = np.mean(x_train)
y_mean = np.mean(y_train)

#Movements#
xi_movement = x_train - x_mean
yi_movement = y_train - y_mean

#Numerator#
z = np.sum(xi_movement * yi_movement)

#Denominator#
q = np.sum(xi_movement **2)

#w#
w = z / q 

#b#
b = y_mean - w * x_mean

#Result#
Y = w * x_train + b

#j#
j_i = (Y - y_train) **2
j_manual = j_i.sum()
j = j_manual / (2 * m)

###Gradient Descent###
def computeGradient(x, y, w, b, m, a, iterations):

    for i in range(iterations):

        Y = w * x + b
   
    #Gradient#
    gradientW = 1 / m * np.sum((Y - y) * x)
    gradientB = 1 / m * (np.sum(Y - y))

    #New#
    newW = w - a * gradientW
    newB = b - a * gradientB

    return float(newW), float(newB)

GradientDescent = computeGradient(x_train, y_train, w, b, m, a, iterations)

print (GradientDescent)