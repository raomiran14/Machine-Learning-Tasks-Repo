import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dataset=pd.read_csv("multiple_linear_regression_dataset.csv")
print(dataset.head(3))
print(dataset.shape)
print(dataset.isnull().sum())
sns.pairplot(data=dataset)
plt.show()
