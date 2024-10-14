import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from itertools import product

df = pd.read_csv('titanic_train.csv')
print(df.isnull().sum())
print('\nЧисло пропущенных значений поля age:', df.isnull().sum()['age'])
print('Доля выживших:', df['survived'].mean())
print('Доля пропущенных значений в рамках каждого признака:\n', df.isnull().sum()/len(df))
columns_to_drop = ['cabin', 'home.dest', 'ticket']
df_droped = df.drop(columns_to_drop, axis=1)
df_droped['fam_size'] = df_droped['sibsp'] + df_droped['parch']
df_droped = df_droped.drop(columns=['sibsp', 'parch'])
print('\nФинальное число признаков:', df_droped.shape[-1]-1)
print('Среднее выборочное fam_size:', df_droped['fam_size'].mean())
classes = [1, 2, 3]
gender = ['female', 'male']
prob_to_survive = {}
for cl, gn in product(classes, gender):
    prob_to_survive['{} & pclass:{}'.format(gn, cl)] = len(df[(df['pclass'] == cl)
                                        & (df['sex'] == gn) & df['survived'] == 1])/len(df[(df['pclass'] == cl) & (df['sex'] == gn)])

print(prob_to_survive)
sns.histplot(data=df_droped[df_droped['survived'] == 0], x = 'age', bins = 20)
plt.show()
sns.histplot(data=df_droped[df_droped['survived'] == 1], x = 'age', bins = 20)
plt.show()

def build_model(df):
    X, y = df.drop('survived', axis=1), df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=13, stratify=y)
    lr = LogisticRegression(random_state=13, max_iter=1000).fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    f_1_score = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    return f_1_score, report

cat_features = ['name', 'sex', 'embarked']
non_cat_features = list(set(df_droped.columns).difference(set(cat_features)))
score, report = build_model(df_droped[non_cat_features].dropna())
print('f1 score модели:', score)

df_imputed_by_mean = df_droped[non_cat_features].fillna(df_droped[non_cat_features].mean()['age'])
score, report = build_model(df_imputed_by_mean)
print('f1_score модели imputed by mean:', score)

df_droped['honorific'] = list(df_droped['name'].str.extract('([A-Za-z]+)\.')[0])
pd.crosstab(df_droped['sex'],df_droped['honorific']).style.background_gradient()
print('Число уникальных honorific:', len(df_droped['honorific'].unique()))
print(df_droped['honorific'].unique())

df_droped['honorific'].\
replace(['Rev', 'Col', 'Dr', 'Major', 'Don', 'Capt', 'Dona', 'Countess', 'Mlle', 'Ms'],
        ['Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Miss', 'Miss'], inplace=True)
pd.crosstab(df_droped['sex'], df_droped['honorific']).style.background_gradient()
print('Доля строк со значением Master:', len(df_droped[df_droped['honorific']=='Master'])/len(df_droped[df_droped['sex']=='male']))
print('Средний возраст категории Mrs:', dict(df_droped.groupby('honorific')['age'].mean()))

df_droped.loc[(df_droped['age'].isnull()) & (df_droped['honorific'] == 'Master'), 'age'] = dict(df_droped.groupby('honorific')['age'].mean())['Master']
df_droped.loc[(df_droped['age'].isnull()) & (df_droped['honorific'] == 'Miss'), 'age'] = dict(df_droped.groupby('honorific')['age'].mean())['Miss']
df_droped.loc[(df_droped['age'].isnull()) & (df_droped['honorific'] == 'Mr'), 'age'] = dict(df_droped.groupby('honorific')['age'].mean())['Mr']
df_droped.loc[(df_droped['age'].isnull()) & (df_droped['honorific'] == 'Mrs'), 'age'] = dict(df_droped.groupby('honorific')['age'].mean())['Mrs']
print(df_droped.isnull().sum())

df_imputed_by_honorific = df_droped.drop(columns=cat_features+['honorific'])
score, report = build_model(df_imputed_by_honorific)
print('f1 score для honorific:', score)

df_droped_one_hot = df_droped.drop(columns=['name', 'honorific'])
df_droped_one_hot = pd.get_dummies(df_droped_one_hot, drop_first=True)
score, report = build_model(df_droped_one_hot)
print('f1 score для one hot:', score, report)