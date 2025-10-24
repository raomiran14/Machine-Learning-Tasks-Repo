import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

dataset = pd.read_csv("Social_Network_Ads.csv")
dataset.drop(columns=["EstimatedSalary"], inplace=True)
print(dataset.head(3))

plt.figure(figsize=(4,3))
sns.scatterplot(x="Age", y="Purchased", data=dataset)
plt.show()
