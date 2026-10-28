import time
from tensorflow.keras.datasets import fashion_mnist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

(Xtr, ytr), (Xte, yte) = fashion_mnist.load_data()
Xtr, ytr = Xtr[:5000].reshape(-1, 784) / 255.0, ytr[:5000]     # subset for speed
Xte, yte = Xte[:1000].reshape(-1, 784) / 255.0, yte[:1000]

for k in ['linear', 'poly', 'rbf']:
    t = time.time()
    p = SVC(kernel=k, degree=3).fit(Xtr, ytr).predict(Xte)
    print(f'{k:7s} Acc: {accuracy_score(yte, p):.4f}  F1 (macro): {f1_score(yte, p, average="macro"):.4f}  '
          f'Time: {time.time() - t:.1f}s')
