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


#if our data is not linearly seperable than we will use polynomial features