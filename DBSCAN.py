import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=250, noise=0.05)

df = {"data1": x[:, 0], "data2": x[:, 1]}
dataset = pd.DataFrame(df)

print(dataset.head(3))

sns.scatterplot(x="data1", y="data2", data=dataset)
plt.show()

from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.2, min_samples=5)
dataset["predict"] = db.fit_predict(dataset)

print(dataset.head(3))