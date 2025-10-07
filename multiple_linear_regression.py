import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dataset=pd.read_csv("multiple_linear_regression_dataset.csv")
print(dataset.head(3))
print(dataset.shape)
print(dataset.isnull().sum())
sns.pairplot(data=dataset)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x = dataset[["age", "experience"]]
y = dataset["income"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test) * 100)
print(lr.predict(x_test))
