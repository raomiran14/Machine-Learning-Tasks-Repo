import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("iris.csv")
print(dataset.head(3))
print(dataset["species"].unique())

sns.pairplot(data=dataset, hue="species")
plt.show()

x = dataset.iloc[:, :-1]
y = dataset["species"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

