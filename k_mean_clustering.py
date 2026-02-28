import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
dataset=pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\iris_modified.csv")
print(dataset.head(3))
sns.pairplot(data=dataset)
#plt.show()
from sklearn.cluster import KMeans
wcss=[]
for i in range(2,21):
    km=KMeans(n_clusters=i,init="k-means++")
    km.fit(dataset)
    wcss.append(km.inertia_)
plt.figure(figsize=(10,5))
plt.plot([i for i in range(2,21)],wcss,marker="o")
plt.xlabel("no of cluster")
plt.xticks([i for i in range(2,21)])
plt.ylabel("wcss")
plt.grid(axis="x")
plt.show()
kmn=KMeans(n_clusters=3)
dataset["predict"]=kmn.fit_predict(dataset)
print(dataset)
sns.pairplot(data=dataset,hue="predict")
plt.show()