from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import Decision_Tree
import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv("drug200.csv")
col_names = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']
data = df
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.2, random_state=41)

classifier = Decision_Tree.DecisionTreeClassifier(
    min_samples_split=3, max_depth=3)
# classifier.fit(X_train,Y_train)

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
