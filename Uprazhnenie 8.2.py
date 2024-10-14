import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from pylab import *
loadings = np.array(pd.read_csv('X_loadings.csv', sep =';', header = None)) #матрица весов
reduced = np.array(pd.read_csv('X_reduced.csv', sep =';', header = None)) #матрица счетов
trans = np.transpose(loadings)
x = reduced.dot(trans)
plt.imshow(x, interpolation='nearest')
plt.show()