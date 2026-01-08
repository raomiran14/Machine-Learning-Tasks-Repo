# regression svr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\placement.csv")
print(dataset.head(3))
print(dataset.isnull().sum())

sns.scatterplot(x="cgpa", y="package", data=dataset)
plt.show()

x = dataset[["cgpa"]]
y = dataset["package"]

from sklearn.model_selection import train_test_split
x_train, x_testr, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

from sklearn.svm import SVR

sv = SVR(kernel="linear")
sv.fit(x_train, y_train)

print(sv.score(x_testr, y_test) * 100)
print(sv.score(x_train, y_train) * 100)

sns.scatterplot(x="cgpa", y="package", data=dataset)
plt.plot(dataset["cgpa"], sv.predict(x), color="red")
plt.show()

