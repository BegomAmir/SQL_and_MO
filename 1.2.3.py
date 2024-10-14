
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x = pd.read_csv('auto-mpg-quiz.csv')
x.hist(column = 'hp')
plt.show()
pie_data = x.groupby(['cyl']).count()
plt.pie(pie_data['mpg'], labels=pie_data.index)
plt.show()
