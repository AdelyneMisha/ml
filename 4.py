import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

models = {'MLE (no prior)': LogisticRegression(C=np.inf, max_iter=5000),
          'MAP L2 (Gaussian)': LogisticRegression(l1_ratio=0.0, C=0.1),
          'MAP L1 (Laplace)': LogisticRegression(l1_ratio=1.0, C=0.1, solver='liblinear')}
for n, m in models.items():
    m.fit(Xtr, ytr)
    w = m.coef_[0]
    print(f'{n:18s} Train: {m.score(Xtr, ytr):.4f}  Test: {accuracy_score(yte, m.predict(Xte)):.4f}  '
          f'Mean|w|: {np.abs(w).mean():.3f}  Zero weights: {(w == 0).sum()}')


    
