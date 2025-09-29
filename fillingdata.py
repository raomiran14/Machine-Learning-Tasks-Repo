import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
dataset=pd.read_csv("Titanic-Dataset.csv")
print(dataset.head(3))
#dataset = dataset.fillna(method="bfill", axis=1)
#dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mode()[0])#this method is used to fill a column using mod usually we fill categorical data with this method but here in this dataset which i practice here is not much categorical data
print(dataset.select_dtypes(include="object").columns)
print(dataset.isnull().sum())
for i in dataset.select_dtypes(include="object").columns:
    dataset[i] = dataset[i].fillna(dataset[i].mode()[0])#this method is used to fill all  categorical(object)data at once
print(dataset.isnull().sum())










