import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('adult.csv', skipinitialspace=True).replace('?', np.nan).dropna()
y = df.iloc[:, -1].str.contains('>50K').astype(int)      
X = pd.get_dummies(df.iloc[:, :-1], drop_first=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)

models = {'Logistic Regression': (LogisticRegression(max_iter=1000), sc.transform(Xtr), sc.transform(Xte)),
          'Decision Tree': (DecisionTreeClassifier(max_depth=5, random_state=0), Xtr, Xte)}
for n, (m, a, b) in models.items():
    p = m.fit(a, ytr).predict(b)
    print(f'{n:20s} Acc: {accuracy_score(yte, p):.4f}  Prec: {precision_score(yte, p):.4f}  '
          f'Rec: {recall_score(yte, p):.4f}  F1: {f1_score(yte, p):.4f}')
