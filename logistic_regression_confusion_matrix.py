import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("placement.csv")
print(dataset.head(3))

x = dataset.iloc[:, :-1]
y = dataset["placed"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(  x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)

print("Train Accuracy:", lr.score(x_train, y_train) * 100)
print("Test Accuracy:", lr.score(x_test, y_test) * 100)

from sklearn.metrics import confusion_matrix

y_pred = lr.predict(x_test)
cm = confusion_matrix(y_test, y_pred)


sns.heatmap(cm, annot=True)
plt.show()