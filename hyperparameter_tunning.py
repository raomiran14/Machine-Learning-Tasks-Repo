import pandas as pd

dataset = pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\polynomial regression.csv")
print(dataset.head(3))

x = dataset.iloc[:, :-1]
y = dataset["Salary"]
