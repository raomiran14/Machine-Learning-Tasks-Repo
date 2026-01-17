import pandas as pd

dataset = pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\placement.csv")
print(dataset.head(3))

x = dataset.iloc[:, :-1]
y = dataset["package"]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

p = cross_val_score(LinearRegression(), x, y,  cv=KFold(n_splits=10))
p.sort()
print(p * 100)

new_data = dataset.head(10)
x_new = new_data.iloc[:, :-1]
y_new = new_data["package"]

from sklearn.model_selection import LeaveOneOut, LeavePOut, KFold, StratifiedKFold

lo = LeaveOneOut()
for train, test in lo.split(x_new, y_new):
    print(train, test)
