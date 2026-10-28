import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

df = pd.read_csv('auto-mpg.csv')[['displacement','mpg']].dropna()
X,y = df[['displacement']].values, df['mpg'].values
xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

xs = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
plt.scatter(X, y, s=10, c='gray')
for d in [1,2,3,5]:
    m = make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression()).fit(xtr, ytr)
    p = m.predict(xte)
    print('Degree', d, 'MSE:', round(mean_squared_error(yte, p), 2), 'R2:', round(r2_score(yte, p), 4))
    plt.plot(xs, m.predict(xs), label=f'degree {d}')
plt.xlabel('Displacement'); plt.ylabel('MPG'); plt.legend(); plt.show()
