import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('asset-v1_ITMOUniversity+INTROMLADVML+fall_2023_ITMO_mag+type@asset+block@pulsar_stars_new (1).csv')
df = df.drop[]