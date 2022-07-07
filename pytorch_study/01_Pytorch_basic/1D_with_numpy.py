import numpy as np

t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)

print('Rank of t : ', t.ndim)       # .ndim function ==  Dimention of tensor
print('Shape of t : ', t.shape)     # .shpe funcition == Size of tensor

# The Output : (7,) <- This means same as (1,7) tensor == (1 x 7) tensor


# 1 Dimension is like List in python

print('t[0] t[1] t[2] = ',t[0],t[1],t[-1])
print(t[0] + t[1])
print(type(t[0])) # type <= numpy.float64 : saved as float type 0. == 0.0

# It can be also sliced [Does not comprise(have) the last index] 

print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1])