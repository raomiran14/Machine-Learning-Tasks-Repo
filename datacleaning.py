import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
dataset=pd.read_csv("Titanic-Dataset.csv")
print(dataset.head(3))
print(dataset.shape)
print(dataset.isnull().sum())#this line tell us missing values in each column 
print((dataset.isnull().sum()/dataset.shape[0])*100)#so in this line we find percentage of null values in each column which make it easy to understand how many null values are present
print(dataset.isnull().sum().sum())#this line tell us total missing values in whole dataset
print((dataset.isnull().sum().sum()/(dataset.shape[0]*dataset.shape[1]))*100)#this line tell us percentage of total null values in whole dataset
dataset.drop(columns=["Cabin"],inplace=True)
print(dataset.isnull().sum())
print(dataset.shape)


# sns.heatmap(dataset.isnull())
# plt.show()