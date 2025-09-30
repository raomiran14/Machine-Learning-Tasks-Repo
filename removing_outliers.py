import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
dataset=pd.read_csv("loan_sanction_train.csv")
print(dataset.head(3))
print(dataset.info())
print(dataset.describe())
# sns.boxplot(x="CoapplicantIncome",data=dataset)
# plt.show()
#now removing ouitlier using direct method
min_range=dataset["CoapplicantIncome"].mean()-(3*dataset["CoapplicantIncome"].std())
max_range=dataset["CoapplicantIncome"].mean()+(3*dataset["CoapplicantIncome"].std())
print(min_range,max_range)
new_data=dataset[dataset["CoapplicantIncome"]<=max_range]
# sns.boxplot(x="CoapplicantIncome",data=new_data)
# plt.show()
#now removing outlier using z-score method
z_score=(dataset["CoapplicantIncome"]-dataset["CoapplicantIncome"].mean())/(dataset["CoapplicantIncome"].std())
dataset["z_score"]=z_score
dataset[dataset["z_score"]<3]
print(dataset)
#so z_score method and direct method both works same