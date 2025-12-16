import numpy as np    # it is an unofficial standard to use np for numpy
import time


a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])


def my_dot(a, b): 
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x
print(f"my_dot(a, b) = {my_dot(a, b)}")


def my_dot2(a, b): 
    return (a * b).sum()
print(f"my_dot(a, b) = {my_dot2(a, b)}")


# test 1-D
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

