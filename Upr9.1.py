import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
dataset = np.array(pd.read_csv('Upr9.1.csv', sep =';'))
x = np.array(dataset[:,0]).reshape((-1,1))
y = np.array(dataset[:,1])
reg = LinearRegression().fit(x, y)
R = reg.score(x, y, sample_weight=None)
print('Выборочное х:', sum(x)/10, '\n'
    'Выборочное у:', sum(y)/10, '\n'                          
    'Коэффициент Тета1:', reg.coef_, '\n'
    'Коэффициент Тета0:', reg.intercept_,'\n'
    'Точность модели:', R)
