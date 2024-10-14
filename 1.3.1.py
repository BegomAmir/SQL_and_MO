import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('1.3.1.csv.')
data.index += 1
data['exp'] = data['y'].ewm(alpha=0.21, adjust=False).mean()
x = data.index.to_numpy()
y = data['y'].to_numpy()
poly = np.polyfit(x,y,1)
a = round(poly[0], 2)
b = round(poly[1], 2)
x = np.arange(1, 101)
data['lin_trend'] = a * x + b
plt.figure(figsize=(20, 8))
plt.plot('y', data=data)
plt.plot('lin_trend', data=data)
plt.xlabel('Time step')
plt.ylabel('y')
plt.show()
f_i = data['lin_trend']
y_avg = data['y'].mean()
R2 = 1 - ((y - f_i) ** 2).sum() / ((y - y_avg) ** 2).sum()
R2 = round(R2, 3)
y_101 = a * 101 + b
y_101 = round(y_101, 0)
print('Коэффициент детерминации:', R2,'\n'
      'Спрогнозированное значение 101 члена ряда:', y_101)
