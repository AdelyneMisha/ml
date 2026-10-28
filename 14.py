import time, matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0                                            
Xtr, ytr, Xte, yte = X[:10000], y[:10000], X[60000:62000], y[60000:62000]   

for act in ['logistic', 'relu', 'tanh']:                 
    t = time.time()
    m = MLPClassifier(hidden_layer_sizes=(64,), activation=act, max_iter=30, random_state=0).fit(Xtr, ytr)
    print(f'{act:9s} Train: {m.score(Xtr, ytr):.4f}  Test: {m.score(Xte, yte):.4f}  '
          f'Time: {time.time() - t:.1f}s  Final loss: {m.loss_:.4f}')
    plt.plot(m.loss_curve_, label=act)
plt.xlabel('Epoch'); plt.ylabel('Training loss'); plt.legend(); plt.show()
