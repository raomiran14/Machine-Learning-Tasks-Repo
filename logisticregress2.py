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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#this code is of logistic regression when we have multiple inputs and are linearly seperated but here the dataset which we use is not linearly seperated but if we use linearly seperated one  it is better 