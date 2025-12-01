import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')


x_train = np.array([1.0, 2.0, 3.0])

y_train = np.array([300.0, 500.0, 600.0])

w = 100
b = 100

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

f_wb = compute_model_output(x_train, w, b)
print(f"{f_wb}")

x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")


plt.plot(x_train, f_wb, c='b',label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

