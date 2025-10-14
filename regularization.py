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
