import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("placement.csv")
print(dataset.head(3))

x = dataset.iloc[:, :-1]
y = dataset["placed"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test) * 100)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

cf = confusion_matrix(y_test, lr.predict(x_test))
print(cf)

sns.heatmap(cf, annot=True)
plt.show()

print(precision_score(y_test, lr.predict(x_test)) * 100)
print(recall_score(y_test, lr.predict(x_test)) * 100)
print(f1_score(y_test, lr.predict(x_test)) * 100)
