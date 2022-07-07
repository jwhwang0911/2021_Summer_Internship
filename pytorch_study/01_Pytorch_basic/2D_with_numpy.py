import numpy as np

t = np.array([
    [1.,2.,3.],
    [4.,5.,6.],
    [7.,8.,9.],
    [10.,11.,12]
])

print('Rank of t : ',t.ndim)
print('Shape of t : ',t.shape)

print('Type of (3,1) : ',type(t[2][0]))
print('Type of (4,4) : ',t[3][2], type(t[3][2])) # Output : Float
# We wanted to Define as Integer, but Python treat as Float
