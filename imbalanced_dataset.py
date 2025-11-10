import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")
print(dataset.head(25))
print(dataset["Purchased"].value_counts())

x = dataset.iloc[:, :-1]
y = dataset["Purchased"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
