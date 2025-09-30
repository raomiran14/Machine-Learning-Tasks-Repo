import pandas as pd
df=pd.DataFrame({"Size":["s","m","l","xl","s","m","l","s","s","l","xl","m"]})
print(df)
ord_data=[["s","m","l","xl"]]
from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder(categories=ord_data)
oe.fit(df[["Size"]])
df["Size_en"]=oe.transform(df[["Size"]])
print(df)

#ordinal encoding with map function
ord_data1={"s":5,"m":6,"l":7,"xl":8}
df["Size_en_map"]=df["Size"].map(ord_data1)
print(df)
#now performing ordinal encoding on original dataset
dataset=pd.read_csv("loan_sanction_train.csv")
print(dataset.head(3))
print(dataset["Property_Area"].unique())#this will give all data in property_area
en_data_ord=[["Rural","Semiurban","Urban"]]
from sklearn.preprocessing import OrdinalEncoder
oen=OrdinalEncoder(categories=en_data_ord)
dataset["Property_Area"]=oen.fit_transform(dataset[["Property_Area"]])
print(dataset)
