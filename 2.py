import pandas as pd

df = pd.read_csv('Rosstat.csv', encoding='utf-8', header=0)
print(df.describe())