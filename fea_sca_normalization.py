import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
dataset=pd.read_csv("loan_sanction_train.csv")
print(dataset.head(3))
print(dataset.isnull().sum())
print(dataset.describe())
from sklearn.preprocessing import MinMaxScaler
ms=MinMaxScaler()
ms.fit(dataset[["CoapplicantIncome"]])
dataset["CoapplicantIncome_min"]=ms.transform(dataset[["CoapplicantIncome"]])
print(dataset.head(3))
print(dataset.describe())
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("before")
sns.histplot(dataset["CoapplicantIncome"], kde=True)
plt.subplot(1,2,2)
plt.title("after")
sns.histplot(dataset["CoapplicantIncome_min"], kde=True)
plt.show()