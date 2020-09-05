#%%
from random import random

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data_file = "csv/breast_cancer/data.csv"
dataset = pd.read_csv(data_file,index_col=0)
dataset['diagnosis'] = pd.array(list(map(lambda x: 0.0 if x == 'M' else 1.0, dataset['diagnosis'])))

del dataset["Unnamed: 32"]
dataset.describe()

X_train, X_test, y_train, y_test = train_test_split(dataset, dataset['diagnosis'], test_size=0.33,
                                                    random_state=random.random()*100)
scores = dict()

for k in [1, 3, 6, 9, 12, 15, 20, 30]:
    knn = KNeighborsClassifier(k)
    knn.fit(X_train, y_train)
    scores[k] = knn.score(X_test, y_test)

plt.plot(scores.keys(), scores.values())
plt.xlabel("K neighbours")
plt.ylabel("accuracy")
plt.title("Scores by K neighbours")
plt.show()

