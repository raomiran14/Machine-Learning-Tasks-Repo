import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
dataset=pd.read_csv("loan_sanction_train.csv")
print(dataset.head(3))
print(dataset.isnull().sum())
print(dataset.describe())
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(dataset[["ApplicantIncome"]])
dataset["ApplicantIncome_ss"]=pd.DataFrame(ss.transform(dataset[["ApplicantIncome"]]),columns=["x"])
print(dataset.head(3))
print(dataset.describe())
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("before")
sns.histplot(dataset["ApplicantIncome"], kde=True)
plt.subplot(1,2,2)
plt.title("after")
sns.histplot(dataset["ApplicantIncome_ss"], kde=True)
plt.show()
#the above technique is standardization
