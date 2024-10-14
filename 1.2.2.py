import pandas as pd
import numpy as np

data = pd.read_csv('Flats.csv', delimiter=',', decimal='.', index_col ='ID')
data = pd.DataFrame(data)
data = data.loc[data['INTERNET'] != 0]
data = 1 - np.exp(1 - data/data.min())
data = data.fillna(0)
data['SUM'] = 0
data['SUM'] = data['DISTANCE'] + data['STOP_COUNT'] + data['COST'] + data['FITNESS'] * 0.5 - data['INTERNET'] * 0.2
x = data.sort_values(by = ['SUM'])
print(x.to_string())
