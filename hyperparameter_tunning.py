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

from sklearn.model_selection import RandomizedSearchCV

df = {
    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    "splitter": ['best', 'random'],
    "max_depth": [i for i in range(2, 20)]
}

rd = RandomizedSearchCV(
    DecisionTreeRegressor(),
    param_distributions=df,
    n_iter=20
)
rd.fit(x_train, y_train)

print(rd.best_params_)
print(rd.best_score_)

from sklearn.model_selection import GridSearchCV

gd = GridSearchCV(
    DecisionTreeRegressor(),
    param_grid=df
)
gd.fit(x_train, y_train)

print(gd.best_params_)
print(gd.best_score_)
