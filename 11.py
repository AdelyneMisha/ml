import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
X, y = X[:, 2:], (y != 0).astype(int)        
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

svm = SVC(kernel='linear', C=1).fit(Xtr, ytr)
print('Test accuracy:', svm.score(Xte, yte))
print('Margin width:', round(2 / np.linalg.norm(svm.coef_[0]), 3))

a, b = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
Z = svm.decision_function(np.c_[a.ravel(), b.ravel()]).reshape(a.shape)
plt.contour(a, b, Z, levels=[-1, 0, 1], colors='k', linestyles=['--', '-', '--'])
plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap='coolwarm')
plt.scatter(*svm.support_vectors_.T, s=120, facecolors='none', edgecolors='g', label='Support vectors')
plt.xlabel('Petal length (scaled)'); plt.ylabel('Petal width (scaled)'); plt.legend(); plt.show()
