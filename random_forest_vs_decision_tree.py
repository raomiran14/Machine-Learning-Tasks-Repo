import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
dataset = pd.read_csv("placement.csv")
print(dataset.head(3))

# features and target
x = dataset.iloc[:, :-1]
y = dataset["placed"]