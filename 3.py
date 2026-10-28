import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

alphas = np.logspace(-3, 2, 50)
models = {'Linear': LinearRegression(),
          'Ridge': RidgeCV(alphas=alphas, cv=5),
          'Lasso': LassoCV(alphas=alphas, cv=5)}
for n, m in models.items():
    p = m.fit(Xtr, ytr).predict(Xte)
    print(n, 'MSE:', round(mean_squared_error(yte, p), 2), 'R2:', round(r2_score(yte, p), 4),
          'alpha:', round(getattr(m, 'alpha_', 0), 4))
print('Lasso zero coefficients:', (models['Lasso'].coef_ == 0).sum())
