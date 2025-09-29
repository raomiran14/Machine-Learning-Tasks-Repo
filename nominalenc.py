import pandas as pd
df=pd.DataFrame({"name":["cow","cat","dog","sheep","goat"]})
print(df)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["en_name"]=le.fit_transform(df["name"])
print(df)
dataset=pd.read_csv("loan_sanction_train.csv")
print(dataset.head(3))
la=LabelEncoder()
la.fit(dataset["Property_Area"])
dataset["Property_Area"]=la.transform(dataset["Property_Area"])
print(dataset)
#this code is basically of label encoding