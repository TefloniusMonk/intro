import statistics
from random import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data_file = "csv/breast_cancer/data.csv"
dataset = pd.read_csv(data_file, index_col=0)
dataset['diagnosis'] = pd.array(list(map(lambda x: 0.0 if x == 'M' else 1.0, dataset['diagnosis'])))

del dataset["Unnamed: 32"]

dataset.describe()

# acc = []
# for i in range(5):
#     X_train, X_test, y_train, y_test = train_test_split(dataset, dataset['diagnosis'], test_size=0.33,
#                                                         random_state=int(random() * 100))
#     knn = KNeighborsClassifier()
#     knn.fit(X_train, y_train)
#     acc.append(knn.score(X_test, y_test))
#     print(classification_report(y_test, knn.predict(X_test)))
#
# print("Mean acc with default settings: {}".format(statistics.mean(acc)))
# #%%
#
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
#
# max_acc = []
# acc_by_K = dict()
#
# for i in range(10):
#     X_train, X_test, y_train, y_test = train_test_split(dataset, dataset['diagnosis'], test_size=0.33,
#                                                         random_state=int(random() * 100))
#     scores = dict()
#
#     for k in range(1, 20):
#         if k not in acc_by_K.keys():
#             acc_by_K[k] = []
#         knn = KNeighborsClassifier(k)
#         knn.fit(X_train, y_train)
#         scores[k] = knn.score(X_test, y_test)
#         acc_by_K[k].append(scores[k])
#     max_acc.append(max(scores.values()))
#     plt.plot(scores.keys(), scores.values())
#     plt.xlabel("K neighbours")
#     plt.ylabel("accuracy")
#     plt.title("Scores by K neighbours, max: " + str(max(scores.values())))
#     plt.show()
# print("Mean acc: " + str(statistics.mean(max_acc)))
#
# mean_for_K = dict()
# for i in acc_by_K:
#     mean_for_K[i] = statistics.mean(acc_by_K[i])
# max_K, max_K_value = "", 0
#
# for key in mean_for_K:
#     if mean_for_K[key] > max_K_value:
#         max_K, max_K_value = key, mean_for_K[key]
# print("Max accuracy for K: {}, value: {}".format(max_K, max_K_value))

from sklearn import preprocessing

max_acc = []
acc_by_K = dict()

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(dataset, dataset['diagnosis'], test_size=0.33,
                                                        random_state=int(random() * 100))
    scores = dict()
    X_scaled_train, X_scaled_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    for k in range(1, 20):
        if k not in acc_by_K.keys():
            acc_by_K[k] = []
        knn = KNeighborsClassifier(k)
        knn.fit(X_scaled_train, y_train)
        scores[k] = knn.score(X_scaled_test, y_test)
        acc_by_K[k].append(scores[k])
    max_acc.append(max(scores.values()))
print("Mean acc: " + str(statistics.mean(max_acc)))
print("Median acc: " + str(statistics.median(max_acc)))

mean_for_K = dict()
for i in acc_by_K:
    mean_for_K[i] = statistics.median(acc_by_K[i])
max_K, max_K_value = "", 0

for key in mean_for_K:
    if mean_for_K[key] > max_K_value:
        max_K, max_K_value = key, mean_for_K[key]
print("Max accuracy for K after scaling : {}, value: {}".format(max_K, max_K_value))
plt.plot(mean_for_K.keys(), mean_for_K.values())
plt.xlabel("K neighbours")
plt.ylabel("Accuracy")
plt.show()
# Mean acc: 0.9949468085106383
# Max accuracy for K after scaling : 8, value: 0.993563829787234
# Much better!