import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import numpy as np

data = pd.read_csv('10.1.csv', sep=';')
data = pd.DataFrame(data)
data = data.drop('Class', axis=1)
data = data.drop('id', axis=1)
data = np.array(data)
Knn = NearestNeighbors(n_neighbors=1, p=2)
Knn.fit(data)
print('Расстояние до ближайшего соседа по евклидовой метрике:', Knn.kneighbors([[67,43]]))
Knn = NearestNeighbors(n_neighbors=3, p=2)
Knn.fit(data)
print('Идентификатры трех ближайших точек:', Knn.kneighbors([[67,43]]))
Knn = NearestNeighbors(n_neighbors=1, p=1)
Knn.fit(data)
print('Расстояние до ближайшего соседа по манхэттеновскому расстоянию:', Knn.kneighbors([[67,43]]))
Knn = NearestNeighbors(n_neighbors=3, p=1)
Knn.fit(data)
print('Идентификатры трех ближайших точек по манхэттену:', Knn.kneighbors([[67,43]]))