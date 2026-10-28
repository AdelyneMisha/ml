import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

d = fetch_california_housing(as_frame=True).frame
d = d[d.AveRooms < 10]                          # remove outliers
X, y = d[['AveRooms']].values, d['MedHouseVal'].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

mu, sd = Xtr.mean(), Xtr.std()                  # standardize feature
f = lambda x: np.c_[np.ones(len(x)), (x - mu) / sd]   # add bias column
A, B = f(Xtr), f(Xte)

w = np.zeros(2)                                 # gradient descent
for _ in range(1000):
    w -= 0.1 * (2 / len(A)) * A.T @ (A @ w - ytr)

wn = np.linalg.inv(A.T @ A) @ A.T @ ytr         # normal equation

for n, v in [('Gradient Descent', w), ('Normal Equation', wn)]:
    p = B @ v
    print(n, v, 'MSE:', round(mean_squared_error(yte, p), 4), 'R2:', round(r2_score(yte, p), 4))

xs = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.scatter(Xte, yte, s=3)
plt.plot(xs, f(xs) @ wn, 'r', label='GD / Normal Eq. line')
plt.xlabel('AveRooms'); plt.ylabel('Median House Value (x $100k)'); plt.legend(); plt.show()
