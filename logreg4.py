import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("iris.csv")
print(dataset.head(3))
print(dataset["species"].unique())

