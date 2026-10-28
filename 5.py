import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

rm = ('headers', 'footers', 'quotes')
cat = ['sci.space']
all_tr = fetch_20newsgroups(subset='train', remove=rm).data
tr = fetch_20newsgroups(subset='train', categories=cat, remove=rm).data
te = fetch_20newsgroups(subset='test', categories=cat, remove=rm).data

cv = CountVectorizer(stop_words='english', max_features=5000).fit(all_tr)
n = np.asarray(cv.transform(tr).sum(0)).ravel()      
t = np.asarray(cv.transform(te).sum(0)).ravel()      
V, N = len(n), n.sum()

def estimate(a):                                     
    return (n + a - 1) / (N + V * (a - 1))

for a in [1, 2, 5, 10, 50]:
    th = estimate(a)
    with np.errstate(divide='ignore'):
        ll = (t[t > 0] * np.log(th[t > 0])).sum() / t.sum()
    print(f'alpha={a:3d} zero-prob words: {(th == 0).sum():4d}  held-out avg log-lik: {ll:.4f}')
