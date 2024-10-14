import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('asset-v1_ITMOUniversity+INTROMLADVML+fall_2023_ITMO_mag+type@asset+block@pulsar_stars_new (1).csv')
df = df[np.logical_and(df['MIP']>= 10, df['MIP']<=100)]
print(df.count())
print(df.MIP.mean())
print(df.MIP.min())
df = df.sort_values('SIP')
X = df.drop('TG', axis=1)
y = df['TG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df['TG'], random_state=33)
print(X['STDC'].max())
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled[:,1].mean())
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
f_1_score = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)
cm = confusion_matrix(y_test, y_pred)
print(report)
print(cm)
lr = LogisticRegression().fit(X_train_scaled, y_train)
y_pred1 = lr.predict(X_test_scaled)
f_1_score1 = f1_score(y_test, y_pred1)
report1 = classification_report(y_test, y_pred1, digits=3)
cm1 = confusion_matrix(y_test, y_pred1)
print(report1)
print(cm1)