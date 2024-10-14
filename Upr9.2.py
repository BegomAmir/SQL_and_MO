import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('fish_train.csv')
X = df.drop('Weight', axis=1)
y = df['Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29, stratify=df['Species'])
print('Выборочное среднее колонки Width:', X_train.Width.mean())
model_basic = LinearRegression().fit(X_train.drop(['Species'], axis=1), y_train)
r2 = r2_score(y_test, model_basic.predict(X_test.drop(['Species'], axis=1)))
print('r2_score полученной модели:', r2)
sns.heatmap(X_train.drop(['Species'], axis=1).corr(), cmap="YlGnBu", annot=True)
plt.show()
Lengths = X_train[['Length1', 'Length2', 'Length3']]
pca = PCA(n_components=3, svd_solver='full').fit(Lengths)
X_train=X_train.drop(['Length1', 'Length2', 'Length3'], axis=1)
X_train['Lengths']=pca.transform(Lengths)[:,0]
print('Доля объясненной дисперсии для тренировочных данных первой ГК:', pca.explained_variance_ratio_[0])
Lengths_test = X_test[['Length1', 'Length2', 'Length3']]
X_test=X_test.drop(['Length1', 'Length2', 'Length3'], axis=1)
X_test['Lengths']=pca.transform(Lengths_test)[:,0]
model_basic=LinearRegression().fit(X_train.drop(['Species'], axis=1), y_train)
r2 = r2_score(y_test, model_basic.predict(X_test.drop(['Species'], axis=1)))
print('r2_score полученной модели с новой колонкой Lengths:', r2)
sns.pairplot(pd.concat([X_train, y_train], axis=1), hue='Species')
plt.show()
X_train_not_cube = X_train.copy()
X_test_not_cube = X_train.copy()
X_train[['Height', 'Width', 'Lengths']] = X_train[['Height', 'Width', 'Lengths']] ** 3
X_test[['Height', 'Width', 'Lengths']] = X_test[['Height', 'Width', 'Lengths']] ** 3
sns.pairplot(pd.concat([X_train, y_train], axis=1), hue='Species')
plt.show()
print('Выборочное среднее колонки Width в кубе:', X_train.Width.mean())
model_basic=LinearRegression().fit(X_train.drop(['Species'], axis=1), y_train)
r2 = r2_score(y_test, model_basic.predict(X_test.drop(['Species'], axis=1)))
print('r2_score полученной модели после возведения в куб:', r2)
X_train_dummies = pd.get_dummies(X_train)
X_test_dummies = pd.get_dummies(X_test)
model_basic=LinearRegression().fit(X_train_dummies, y_train)
r2 = r2_score(y_test, model_basic.predict(X_test_dummies))
print('r2_score полученной модели после кодировки:', r2)
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
model_basic = LinearRegression().fit(X_train_dummies, y_train)
r2 = r2_score(y_test, model_basic.predict(X_test_dummies))
print('r2_score полученной модели после кодировки и внесения категориальных признаков:', r2)