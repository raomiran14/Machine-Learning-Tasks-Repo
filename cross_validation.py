import pandas as pd

dataset = pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\placement.csv")
print(dataset.head(3))

x = dataset.iloc[:, :-1]
y = dataset["package"]
