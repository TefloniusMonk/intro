# %% Load dataset

import pandas as pd

CROSS_VALIDATION_ITER = 25

data_file = "csv/breast_cancer/data.csv"
dataset = pd.read_csv(data_file, index_col=0)
dataset['diagnosis'] = pd.array(list(map(lambda x: 0.0 if x == 'M' else 1.0, dataset['diagnosis'])))

del dataset["Unnamed: 32"]

dataset.describe()

print(dataset.corr().abs())

corr_matrix = dataset.corr().abs()
do_not_drop = set()
for row in corr_matrix.columns:
    for column in corr_matrix.columns:
        if column == row:
            continue
        if corr_matrix[row][column] > 0.85:
            if column in dataset and column not in do_not_drop:
                del dataset[column]
                do_not_drop.add(row)

print(dataset)
