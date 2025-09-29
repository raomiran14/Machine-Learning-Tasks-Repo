import pandas as pd
dataset=pd.read_csv("loan_sanction_train.csv")
print(dataset.head(3))
print(dataset.isnull().sum())
dataset["Gender"]=dataset["Gender"].fillna(dataset["Gender"].mode()[0])
dataset["Married"]=dataset["Married"].fillna(dataset["Married"].mode()[0])
print(dataset.isnull().sum())
en_data=dataset[["Gender","Married"]]
print(en_data)
result=pd.get_dummies(en_data).info()
print(result)#pd.getdummies is a pandas way to do encoding but it is not good because we have to do more steps to do encoding so scikit learn way onehotencoder is best
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(drop="first")
ar=ohe.fit_transform(en_data).toarray()
print(ar)
dataframe=pd.DataFrame(ar,columns=["Gender_Male","Married_Yes"])
print(dataframe)


