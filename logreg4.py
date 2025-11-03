import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("iris.csv")
print(dataset.head(3))
print(dataset["species"].unique())

sns.pairplot(data=dataset, hue="species")
plt.show()

x = dataset.iloc[:, :-1]
y = dataset["species"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# One-vs-Rest (OVR)
lr = LogisticRegression(multi_class="ovr")
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test) * 100)

# Multinomial
lr1 = LogisticRegression(multi_class="multinomial")
lr1.fit(x_train, y_train)
print(lr1.score(x_test, y_test) * 100)


