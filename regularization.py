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

