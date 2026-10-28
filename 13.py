import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv('winequality-red.csv', sep=';').dropna()
print(df.describe())
X, y = df.drop('quality', axis=1), (df.quality >= 6).astype(int)   
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

for arch in [(8,), (32,), (64, 32), (128, 64, 32)]:
    m = MLPClassifier(hidden_layer_sizes=arch, max_iter=1000, random_state=0).fit(Xtr, ytr)
    p = m.predict(Xte)
    print(f'{str(arch):15s} Train: {m.score(Xtr, ytr):.4f}  Test: {accuracy_score(yte, p):.4f}  '
          f'F1: {f1_score(yte, p):.4f}')
#some error in dataset note and check
