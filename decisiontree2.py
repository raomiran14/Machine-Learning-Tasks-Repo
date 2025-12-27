# decision tree regressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\multiple_linear_regression_dataset.csv")
print(dataset.head(3))

sns.pairplot(data=dataset)
# plt.show()

print(dataset.isnull().sum())
