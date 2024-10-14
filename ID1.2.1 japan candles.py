import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

data = pd.read_csv('SPFB.RTS-12.18_180901_181231.csv', encoding='utf-8')
data['DATE'] = data['<DATE>'] + ' ' + data['<TIME>']
data['DATE'] = pd.to_datetime(data['DATE'], format="%d/%m/%y %H:%M")
df_filtered = data[data['DATE'].dt.strftime('%Y-%m-%d') == '2018-09-06']
df_filtered = df_filtered.set_index('DATE')
df_filtered = df_filtered[['<OPEN>','<HIGH>','<LOW>','<CLOSE>']]
df_filtered.columns =['open', 'high', 'low', 'close']
d2 = df_filtered.resample('1H').agg({'open':'first',
                                     'high':'max',
                                     'low':'min',
                                     'close':'last'})
fig = go.Figure(data=[go.Candlestick(x=d2.index,
                open=d2['open'],
                high=d2['high'],
                low=d2['low'],
                close=d2['close'])])

fig.show()