import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("placement.csv")
print(dataset.head(3))

plt.figure(figsize=(5,4))
sns.scatterplot(x="cgpa", y="placement_exam_marks", data=dataset, hue="placed")
plt.show()

x = dataset.iloc[:, :-1]
y = dataset["placed"]

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=3)
pf.fit(x)
x = pd.DataFrame(pf.transform(x))
print(x)


#if our data is not linearly seperable than we will use polynomial features