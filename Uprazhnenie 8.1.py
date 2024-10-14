import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

dataset = np.array(pd.read_csv('artem.csv', header=None))
centered_matrix = scale(dataset, with_mean=True, with_std=False, axis=0)
pca = PCA(n_components=2, svd_solver='full')
x = pca.fit_transform(centered_matrix)
plt.scatter(x[:, 0], x[:, 1])
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.show()
print(x)
print(pca.components_)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
