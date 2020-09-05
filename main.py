#%% Load dataset

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CROSS_VALIDATION_ITER = 25

data_file = "csv/breast_cancer/data.csv"
dataset = pd.read_csv(data_file,index_col=0)
dataset['diagnosis'] = pd.array(list(map(lambda x: 0.0 if x == 'M' else 1.0, dataset['diagnosis'])))

del dataset["Unnamed: 32"]

dataset.describe()

ax = sns.PairGrid(dataset.iloc[:, 0:11], hue="diagnosis", palette="Set2")
ax = ax.map_diag(plt.hist, edgecolor="w")
ax = ax.map_offdiag(plt.scatter, edgecolor="w", s=40)
plt.show()
