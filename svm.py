import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

dataset = pd.read_csv("D:\PROJECTS\Machine-Learning-Tasks-Repo\placement.csv")
print(dataset.head(3))
print(dataset.isnull().sum())

sns.scatterplot(x="cgpa", y="placement_exam_marks",data=dataset, hue="placed")
# plt.show()

x = dataset.iloc[:, :-1]
y = dataset["placed"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=10)

from sklearn.svm import SVC

sv = SVC(kernel="rbf")
sv.fit(x_train, y_train)

print(sv.score(x_test, y_test) * 100)
print(sv.score(x_train, y_train) * 100)

plot_decision_regions(x.to_numpy(), y.to_numpy(), clf=sv)
plt.show()

