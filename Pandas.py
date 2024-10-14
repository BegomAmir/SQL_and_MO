import pandas as pd

df = pd.read_csv('Dataset.csv', encoding='utf-8', header=0)
print(df.describe())
print(df.to_string())
