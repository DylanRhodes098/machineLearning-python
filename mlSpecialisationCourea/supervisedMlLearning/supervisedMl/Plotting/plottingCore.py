import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])

y_train = np.array([300.0, 500.0])

i = 0 

x_i = x_train[i]
y_i = y_train[i]

# Plot the data points
plt.scatter(x_train, y_train, marker='o', c='b')

# Set the title
plt.title("Housing Prices")

# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')

# Set the x-axis label
plt.xlabel('Size (1000 sqft)')

plt.show()