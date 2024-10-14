import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

x = pd.read_csv('1.2.2.csv', sep=';')
df = pd.DataFrame(x)
df = df.fillna('Не выбрано')

#Построение столбчатой диаграммы
FL = list(df['Choice_1'])
counts = Counter(FL)
del counts['Не выбрано']
counts = dict(sorted(counts.items()))
plt.bar(counts.keys(), counts.values())
plt.ylim(0, 80)
plt.xticks(rotation=45)
plt.show()
print(counts)

#Построение круговой диаграммы
dictionary = df.to_dict()
K1 = list(dictionary['Choice_1'].values())
K2 = list(dictionary['Choice_2'].values())
K3 = list(dictionary['Choice_3'].values())
K4 = list(dictionary['Choice_4'].values())
K5 = list(dictionary['Choice_5'].values())
K6 = list(dictionary['Choice_6'].values())
K7 = list(dictionary['Choice_7'].values())
K8 = list(dictionary['Choice_8'].values())
K9 = list(dictionary['Choice_9'].values())
KD = K1 + K2 + K3 + K4 + K5 + K6 + K7 + K8 + K9
L = {}
for element in KD:
    if element in L:
        L[element] += 1
    else:
        L[element] = 1
del L['Не выбрано']
labels = L.keys()
sizes = L.values()
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.show()
print(L)

