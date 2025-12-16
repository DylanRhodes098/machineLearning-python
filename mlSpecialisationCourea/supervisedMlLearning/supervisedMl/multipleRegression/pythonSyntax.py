import numpy as np    # it is an unofficial standard to use np for numpy
import time

a = np.zeros(4);                

a = np.zeros((4,));            

a = np.random.random_sample(4); 

#vector indexing operations on 1-D vectors
a = np.arange(10)

print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

print(f"a[-1] = {a[-1]}")

    #vector slicing operations
a = np.arange(10)
print(f"a         = {a}")

#access 5 consecutive elements (start:stop:step)
c = a[2:7:1];     print("a[2:7:1] = ", c)

# access 3 elements separated by two 
c = a[2:7:2];     print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a[3:];        print("a[3:]    = ", c)

# access all elements below index 3
c = a[:3];        print("a[:3]    = ", c)

# access all elements
c = a[:];         print("a[:]     = ", c)