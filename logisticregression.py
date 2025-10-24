import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

dataset = pd.read_csv("Social_Network_Ads.csv")
dataset.drop(columns=["EstimatedSalary"], inplace=True)
print(dataset.head(3))

plt.figure(figsize=(4,3))
sns.scatterplot(x="Age", y="Purchased", data=dataset)
plt.show()

x = dataset[["Age"]]
y = dataset["Purchased"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test) * 100)
print(lr.predict([[40]]))
