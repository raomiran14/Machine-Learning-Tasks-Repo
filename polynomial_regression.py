import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("polynomial regression.csv")
print(dataset.head(9))
plt.scatter(dataset["Level"], dataset["Salary"])
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
print(dataset.corr())
x = dataset[["Level"]]
y = dataset["Salary"]

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
pf.fit(x)
x = pf.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test))
y_prd = lr.predict(x)
plt.scatter(dataset["Level"], dataset["Salary"])
plt.plot(dataset["Level"], y_prd, c="red")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend(["org", "prd"])
plt.show()

test = pf.transform([[8]])
print(lr.predict(test))

