import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("placement.csv")
print(dataset.head(3))

sns.scatterplot(x="cgpa", y="placement_exam_marks", data=dataset)
plt.show()

x = dataset[["cgpa", "placement_exam_marks"]]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
dataset["cluster"] = kmeans.fit_predict(x)

print(dataset.head(5))


centers = kmeans.cluster_centers_

sns.scatterplot(
    x="cgpa",
    y="placement_exam_marks",
    data=dataset,
    hue="cluster"
)

plt.scatter(centers[:, 0], centers[:, 1], s=100, c='red', label="centroids")

plt.legend()
plt.show()