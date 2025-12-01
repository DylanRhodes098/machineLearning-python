import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([[1.0, 2.0],
                   [1.0, 2.0],
                   [1.0, 2.0]])
y_train = np.array([300.0, 500.0])
m = x_train.shape[1]
m2 = len(x_train)

i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
