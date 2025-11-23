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

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gnb = GaussianNB()
gnb.fit(x_train, y_train)

print(gnb.score(x_test, y_test) * 100), print(gnb.score(x_train, y_train) * 100)

plot_decision_regions(x.to_numpy(), y.to_numpy(), clf=gnb)
plt.show()

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
print(mnb.score(x_test, y_test) * 100), print(mnb.score(x_train, y_train) * 100)
plot_decision_regions(x.to_numpy(), y.to_numpy(), clf=mnb)
plt.show()

bnb = BernoulliNB()
bnb.fit(x_train, y_train)
print(mnb.score(x_test, y_test) * 100), print(mnb.score(x_train, y_train) )
plot_decision_regions(x.to_numpy(), y.to_numpy(), clf=bnb)
plt.show()

# in my case dataset is not linearly separable so use linearly separable data for better accuracy
