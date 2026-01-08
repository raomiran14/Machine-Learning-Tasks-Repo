# regression svr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\placement.csv")
print(dataset.head(3))
print(dataset.isnull().sum())

sns.scatterplot(x="cgpa", y="package", data=dataset)
plt.show()
