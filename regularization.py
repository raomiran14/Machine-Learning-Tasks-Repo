import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Step 1: Load the dataset
dataset = pd.read_csv("house_price_regression_dataset.csv") 
print(dataset.head(3))      

# Step 2: Visualize feature correlation
plt.figure(figsize=(10,10))
sns.heatmap(data=dataset.corr(), annot=True)
plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 3: Split into features and target
x = dataset.iloc[:, :-1]
y = dataset["House_Price"]

# Step 4: Scale the features for better model performance
sc = StandardScaler()
sc.fit(x)
x = pd.DataFrame(sc.transform(x), columns=x.columns)
print(x.head())

# Step 5: Train-test split (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Step 6: Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test) * 100)

# Step 7: Evaluate model performance
print("MSE:", mean_squared_error(y_test, lr.predict(x_test)))
print("MAE:", mean_absolute_error(y_test, lr.predict(x_test)))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr.predict(x_test))))

# Step 8: Plot feature coefficients
plt.figure(figsize=(12,5))
plt.bar(x.columns, lr.coef_)
plt.title("Linear Regression Coefficients")
plt.xlabel("columns")
plt.ylabel("coef")
plt.show()

from sklearn.linear_model import Lasso, Ridge

# Step 9: Lasso Regression (L1)
la = Lasso(alpha=10)
la.fit(x_train, y_train)
print("Lasso R2:", la.score(x_test, y_test) * 100)
print("Lasso MSE:", mean_squared_error(y_test, la.predict(x_test)))
print("Lasso MAE:", mean_absolute_error(y_test, la.predict(x_test)))
print("Lasso RMSE:", np.sqrt(mean_squared_error(y_test, la.predict(x_test))))

plt.figure(figsize=(12,5))
plt.bar(x.columns, la.coef_)
plt.title("Lasso Regression Coefficients")
plt.xlabel("columns")
plt.ylabel("coef")
plt.show()

# Step 10: Ridge Regression (L2)
ri = Ridge(alpha=10)
ri.fit(x_train, y_train)
print("Ridge R2:", ri.score(x_test, y_test) * 100)
print("Ridge MSE:", mean_squared_error(y_test, ri.predict(x_test)))
print("Ridge MAE:", mean_absolute_error(y_test, ri.predict(x_test)))
print("Ridge RMSE:", np.sqrt(mean_squared_error(y_test, ri.predict(x_test))))

plt.figure(figsize=(12,5))
plt.bar(x.columns, ri.coef_)
plt.title("Ridge Regression Coefficients")
plt.xlabel("columns")
plt.ylabel("coef")
plt.show()


df = pd.DataFrame({
    "col_name": x.columns,
    "linearregression": lr.coef_,
    "lasso": la.coef_,
    "ridge": ri.coef_
})
print(df)

