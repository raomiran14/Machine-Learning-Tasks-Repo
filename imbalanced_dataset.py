import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")
print(dataset.head(25))
print(dataset["Purchased"].value_counts())

x = dataset.iloc[:, :-1]
y = dataset["Purchased"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test) * 100)
print(lr.predict([[45, 26000]]))

# Note: Dataset is imbalanced — more 0s (257) than 1s (143), 
# so predictions are biased toward 0.
