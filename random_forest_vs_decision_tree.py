import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
dataset = pd.read_csv("placement.csv")
print(dataset.head(3))

# features and target
x = dataset.iloc[:, :-1]
y = dataset["placed"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(  x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

lr = LogisticRegression()
dt = DecisionTreeClassifier()
sv = SVC()

lr.fit(x_train, y_train)
dt.fit(x_train, y_train)
sv.fit(x_train, y_train)

print("LR:", lr.score(x_test, y_test) * 100)
print("DT:", dt.score(x_test, y_test) * 100)
print("SVM:", sv.score(x_test, y_test) * 100)

from sklearn.ensemble import VotingClassifier

models = [ ("lr", LogisticRegression()),("dt", DecisionTreeClassifier()),("sv", SVC())]
vc = VotingClassifier(models)
vc.fit(x_train, y_train)

print("Voting Classifier:", vc.score(x_test, y_test) * 100)