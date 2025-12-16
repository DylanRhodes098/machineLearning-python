import numpy as np    # it is an unofficial standard to use np for numpy
import time

# NumPy routines which allocate memory and fill with user specified values
a3 = np.array([[5], [4], [3]]);   

print(f" a shape = {a3.shape}, np.array: a = {a3}")

#vector indexing operations on matrices
a1 = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
print(f"a.shape: {a1.shape}, \na= {a1}")

#access an element
print(f"\na[2,0].shape:   {a1[2, 0].shape}, a[2,0] = {a1[2, 0]},     type(a[2,0]) = {type(a1[2, 0])} Accessing an element returns a scalar\n")

#access a row
print(f"{a1[2].shape}, {a1[2]},{type(a1[2])}")

#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")

