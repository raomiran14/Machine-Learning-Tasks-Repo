# decision tree regressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\multiple_linear_regression_dataset.csv")
print(dataset.head(3))

sns.pairplot(data=dataset)
# plt.show()

print(dataset.isnull().sum())

x = dataset.iloc[:, :-1]
y = dataset["income"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor, plot_tree

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

print(dt.score(x_test, y_test) * 100)
print(dt.score(x_train, y_train) * 100)

plot_tree(dt)
plt.show()
