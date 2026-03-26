# MAX VOTING CLASSIFIER
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=1000, noise=0.2)

df = {"x1": x[:, 0], "x2": x[:, 1], "y": y}
dataset = pd.DataFrame(df)
print(dataset)

sns.scatterplot(x="x1", y="x2", data=dataset, hue="y")
plt.show()

x_a = dataset.iloc[:, :-1]
y_a = dataset["y"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_a, y_a, test_size=0.2, random_state=42)

