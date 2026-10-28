import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('pima-indians-diabetes.csv')
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan)               
df[cols] = df[cols].fillna(df[cols].median())
X, y = df.drop('Outcome', axis=1), df['Outcome']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)

for name, a, b in [('Without scaling', Xtr, Xte), ('With scaling', sc.transform(Xtr), sc.transform(Xte))]:
    p = LogisticRegression(max_iter=1000).fit(a, ytr).predict(b)
    print(f'{name:16s} Acc: {accuracy_score(yte, p):.4f}  Prec: {precision_score(yte, p):.4f}  '
          f'Rec: {recall_score(yte, p):.4f}  F1: {f1_score(yte, p):.4f}')
