import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

dataset = pd.read_csv("placement.csv")
print(dataset.head(3))
print(dataset.isnull().sum())

# we use kdeplot to see data is normally distributed or not
sns.kdeplot(data=dataset["placement_exam_marks"])
plt.show()

plt.figure(figsize=(4,3))
# we use scatterplot to see is data linearly separable or not
sns.scatterplot(x="cgpa", y="placement_exam_marks", data=dataset, hue="placed")
plt.show()

x = dataset.iloc[:, :-1]
y = dataset["placed"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
