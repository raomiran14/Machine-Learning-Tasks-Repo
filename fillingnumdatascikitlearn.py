import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
dataset=pd.read_csv("Titanic-Dataset.csv")
print(dataset.head(3))
print(dataset.isnull().sum())
dataset.info()
print(dataset.select_dtypes(include=["int64","float64"]).columns)
from sklearn.impute import SimpleImputer 
si=SimpleImputer(strategy="mean")
result=si.fit_transform(dataset[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
print(result)
new_dataset=pd.DataFrame(result,columns=dataset.select_dtypes(include=["int64","float64"]).columns)
print(new_dataset)
print(new_dataset.isnull().sum())
