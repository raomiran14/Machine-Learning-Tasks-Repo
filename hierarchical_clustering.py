import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\iris_modified.csv")
print(dataset.head(2))

sns.pairplot(data=dataset)
plt.show()

import scipy.cluster.hierarchy as sc

sc.dendrogram( sc.linkage(dataset, method="single", metric="euclidean"))
plt.show()

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=2, linkage="single")

dataset["predict"] = ac.fit_predict(dataset)
print(dataset.head(2))

sns.pairplot(data=dataset, hue="predict")
plt.show()