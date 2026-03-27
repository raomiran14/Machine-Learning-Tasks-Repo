import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("placement.csv")
print(dataset.head(3))

x = dataset.iloc[:, :-1]
y = dataset["package"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(  x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.score(x_train, y_train) * 100), print(lr.score(x_test, y_test) * 100)

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
print(dt.score(x_train, y_train) * 100), print(dt.score(x_test, y_test) * 100)

sv = SVR()
sv.fit(x_train, y_train)
print(sv.score(x_train, y_train) * 100), print(sv.score(x_test, y_test) * 100)