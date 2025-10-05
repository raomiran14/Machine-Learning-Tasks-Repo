import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
dataset=pd.read_csv("placement.csv")
print(dataset.head(3))
plt.figure(figsize=(5,3))
sns.scatterplot(x="cgpa",y="package",data=dataset)
plt.show()
print(dataset.isnull().sum())
x=dataset[["cgpa"]]
y=dataset["package"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression 
lr=LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test)*100)
print(lr.predict([[ 6.89]]))
y_prd=lr.predict(x)
plt.figure(figsize=(5,4))
sns.scatterplot(x="cgpa",y="package",data=dataset)
plt.plot(dataset["cgpa"],y_prd,c="red")
plt.legend(["org","predict line"])
plt.show()