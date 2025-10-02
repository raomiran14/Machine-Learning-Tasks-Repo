import pandas as pd
data={
    "name":["a","b","c","d","a","c"],
    "eng":[8,7,5,8,8,5],
    "urdu":[2,3,4,5,2,6]

}
df=pd.DataFrame(data)
print(df)
# df["duplicated"]=df.duplicated()
# print(df)
df.drop_duplicates(inplace=True)
print(df)
#now performing this on real dataset
dataset=pd.read_csv("loan_sanction_train.csv")
print(dataset.head(3))
print(dataset.shape)
dataset.drop_duplicates(inplace=True)
print(dataset.shape)