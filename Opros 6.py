import numpy as np

a = [[3.17],
      [-1.58],
      [-1.59]]
b = [[0.32],
     [0.95]]
d = np.trans pose(b)
c = np.dot(a,d)
print(c)