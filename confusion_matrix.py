import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("placement.csv")
print(dataset.head(3))

x = dataset.iloc[:, :-1]
y = dataset["placed"]
