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

