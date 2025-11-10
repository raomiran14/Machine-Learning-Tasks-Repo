import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")
print(dataset.head(25))
print(dataset["Purchased"].value_counts())
