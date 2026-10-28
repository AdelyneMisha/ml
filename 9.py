import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1', parse_dates=['InvoiceDate'])
df = df.dropna(subset=['CustomerID'])
df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]  
df['Spend'] = df.Quantity * df.UnitPrice
now = df.InvoiceDate.max()

c = df.groupby('CustomerID').agg(Frequency=('InvoiceNo', 'nunique'), Quantity=('Quantity', 'sum'),
                                 Products=('StockCode', 'nunique'), AvgPrice=('UnitPrice', 'mean'),
                                 Recency=('InvoiceDate', lambda x: (now - x.max()).days),
                                 Spend=('Spend', 'sum'))
c['Segment'] = (c.Spend > c.Spend.median()).astype(int)  
X, y = c.drop(['Spend', 'Segment'], axis=1), c.Segment
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0).fit(Xtr, ytr)
print('Test accuracy:', round(tree.score(Xte, yte), 4))
print(pd.Series(tree.feature_importances_, X.columns).round(3))

plt.figure(figsize=(14, 6))
plot_tree(tree, feature_names=X.columns, class_names=['Low', 'High'], filled=True)
plt.show()
