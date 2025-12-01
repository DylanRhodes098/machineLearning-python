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

#Result#
Y = w * x_train + b

#j#
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

#Gradient Descent#
a = 0.01
iterations=1000

for i in range(iterations):

    Y = w * x_train + b
    a = 0.01

    #Gradient#
    gradientW = 1 / m * np.sum((Y - y_train) * x_train)
    gradientB = 1 / m * (np.sum(Y - y_train))

    #New#
    newW = w - a * gradientW
    newB = b - a * gradientB

print(newW)
print(newB)

def compute_gradient(x, y, w, b): 
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing


