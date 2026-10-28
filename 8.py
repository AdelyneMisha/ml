import time
from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

(Xtr, ytr), (Xte, yte) = fashion_mnist.load_data()
Xtr, ytr = Xtr[:10000].reshape(-1, 784) / 255.0, ytr[:10000]   
Xte, yte = Xte[:2000].reshape(-1, 784) / 255.0, yte[:2000]

for k in [1, 3, 5, 7, 9, 15]:
    t = time.time()
    p = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).predict(Xte)
    print(f'K={k:2d}  Accuracy: {accuracy_score(yte, p):.4f}  Time: {time.time() - t:.2f}s')
