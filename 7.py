from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

rm = ('headers', 'footers', 'quotes')
tr = fetch_20newsgroups(subset='train', remove=rm)
te = fetch_20newsgroups(subset='test', remove=rm)
cv = CountVectorizer(stop_words='english', max_features=5000)
Xtr, Xte = cv.fit_transform(tr.data), cv.transform(te.data)

for name, m in [('Multinomial NB', MultinomialNB()), ('Bernoulli NB', BernoulliNB())]:
    p = m.fit(Xtr, tr.target).predict(Xte)
    print(f'{name:15s} Acc: {accuracy_score(te.target, p):.4f}  '
          f'F1 (macro): {f1_score(te.target, p, average="macro"):.4f}')
