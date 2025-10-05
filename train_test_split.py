import pandas as pd
dataset=pd.read_csv("HousingData.csv")
print(dataset.head(3))
print(dataset.shape)
input_data=dataset.iloc[:,:-1]
output_data=dataset["MEDV"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(input_data,output_data,test_size=0.25)
print(y_train)
print(x_train.shape),print(y_train.shape)
print(x_test.shape),print(y_test.shape)
