import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\iris_modified.csv")
print(dataset.head(2))

sns.pairplot(data=dataset)
plt.show()

