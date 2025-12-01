import numpy as np
import matplotlib.pyplot as plt

from lab_utils_uni import (
    plt_intuition,
    plt_stationary,
    plt_update_onclick,
    soup_bowl
)

plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430, 630, 730,])
m = x_train.shape[0] 

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


Y = w * x_train + b

j_i = (Y - y_train) **2
j_manual = j_i.sum()
j = j_manual / (2 * m)

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

total_cost = compute_cost(x_train, y_train, w, b)

print(total_cost)
print (j)
print (w)

plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

