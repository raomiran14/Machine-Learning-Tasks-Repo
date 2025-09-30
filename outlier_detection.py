import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
dataset=pd.read_csv("loan_sanction_train.csv")
print(dataset.head(3))
print(dataset.info())
print(dataset.describe())
print(dataset.shape)
# sns.boxplot(x="CoapplicantIncome",data=dataset)
# plt.show()
#now removing outliers using IQR method
q1=dataset["CoapplicantIncome"].quantile(0.25)
q3=dataset["CoapplicantIncome"].quantile(0.75)
IQR=q3-q1
print(IQR)
min_range=q1-(1.5*IQR)
max_range=q3+(1.5*IQR)
print(min_range,max_range)
new_dataset=dataset[dataset["CoapplicantIncome"]<=max_range]
print(new_dataset.shape)
sns.boxplot(x="CoapplicantIncome",data=new_dataset)
plt.show()

