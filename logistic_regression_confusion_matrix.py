import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("placement.csv")
print(dataset.head(3))

x = dataset.iloc[:, :-1]
y = dataset["placed"]