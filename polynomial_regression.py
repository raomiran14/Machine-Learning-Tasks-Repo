import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("polynomial regression.csv")
print(dataset.head(9))
plt.scatter(dataset["Level"], dataset["Salary"])
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
print(dataset.corr())
