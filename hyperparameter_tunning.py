import pandas as pd

dataset = pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\polynomial regression.csv")
print(dataset.head(3))

x = dataset.iloc[:, :-1]
y = dataset["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(
    criterion='squared_error',
    splitter='best',
    max_depth=2
)
dt.fit(x_train, y_train)

print(dt.score(x_train, y_train) * 100)
print(dt.score(x_test, y_test) * 100)
