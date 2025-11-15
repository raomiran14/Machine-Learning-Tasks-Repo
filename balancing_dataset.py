import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")
print(dataset.head(25))
print(dataset["Purchased"].value_counts())

x = dataset.iloc[:, :-1]
y = dataset["Purchased"]

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

ro = RandomOverSampler()
ro_x, ro_y = ro.fit_resample(x, y)
print(ro_y.value_counts())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(ro_x, ro_y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test) * 100)
print(lr.predict([[29, 80000]]))

ru = RandomUnderSampler()
ru_x, ru_y = ru.fit_resample(x, y)
print(ru_y.value_counts())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(ru_x, ru_y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test) * 100)
print(lr.predict([[29, 80000]]))
