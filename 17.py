import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = load_digits().data / 16          # preprocessing: scale pixels (0-16) to 0-1
ks = range(2, 16)
inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=0).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    print(k, round(km.inertia_, 1), round(sil[-1], 4))
print("Best K:", ks[sil.index(max(sil))])

plt.subplot(121); plt.plot(ks, inertia, 'o-'); plt.title('Elbow (Inertia)'); plt.xlabel('K')
plt.subplot(122); plt.plot(ks, sil, 'o-r'); plt.title('Silhouette'); plt.xlabel('K')
plt.show()
