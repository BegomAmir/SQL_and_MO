import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('adult_data_train.csv')
df = df.drop(['education', 'marital-status'], axis=1)
cat_features = ['workclass', 'occupation', 'relationship','race', 'sex',
                'native-country' ]
print('Количество числовых признаков:', df.select_dtypes(include=['int64']).shape[1]-1)
print('Количество нечисловых признаков:', len(cat_features))
column = 'label'
plt.gcf().set_size_inches(10,7)
sns.barplot(x=df[[column]].groupby(df[column]).count().index, y=df[column].groupby(df[column]).count().to_numpy())
plt.show()
print('Доля объектов класса 0:', len(df[df[column] == 0])/len(df))
df_only_numeric = df.select_dtypes(include=['int64'])

def build_model(df):
  X, y = df.drop('label', axis=1), df['label']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)
  knn = KNeighborsClassifier().fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  f_1_score = f1_score(y_test, y_pred)
  return X_train.fnlwgt.mean(), f_1_score, classification_report(y_test, y_pred, digits=3)

mean, score, report = build_model(df_only_numeric)
print('Выборочное среднее столбца fnlwgt:', mean)
print('F_1(weighted) модели:', score)
print('Отчет о классификации: \n', report)

def build_model(df):
  X, y = df.drop('label', axis=1), df['label']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)
  scaler = MinMaxScaler().fit(X_train)
  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  knn = KNeighborsClassifier().fit(X_train_scaled, y_train)
  y_pred = knn.predict(X_test_scaled)
  f_1_score = f1_score(y_test, y_pred)
  return X_train_scaled[:,1].mean(), f_1_score

mean, score = build_model(df_only_numeric)
print('Выборочное среднее столбца fnlwgt:', mean)
print('F_1(weighted) модели:', score)
import os

if not os.path.exists('pictures'):
  os.makedirs('pictures')

df_cat  = df[cat_features]
for i, column in enumerate(cat_features):
  plt.figure(i)
  plt.gcf().set_size_inches(10, 7)
  sns.barplot(x=df_cat[[column]].groupby(df_cat[column]).count().index, y=df_cat[column].groupby(df_cat[column]).count().to_numpy())
  plt.xticks(rotation=90)
  plt.show()
  print(column)

df_drop_nans = df.replace('?', np.nan)
df_drop_nans = df_drop_nans.dropna()
print('Число строк, содержащих хотя бы один пропуск:', len(df)-len(df_drop_nans))
df_drop_nans_dummies = pd.get_dummies(df_drop_nans, drop_first=True)
print('Число полученных признаков:', df_drop_nans_dummies.shape[1] - 1)