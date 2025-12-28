import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

dataset = pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\Social_Network_Ads.csv")
print(dataset.head(3))

sns.scatterplot(x="Age", y="EstimatedSalary", data=dataset, hue="Purchased")
plt.show()

print(dataset.isnull().sum())
