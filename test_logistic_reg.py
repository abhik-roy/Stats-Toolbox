import numpy as np
import pandas as pd
from statsbox.logistic_regression import LogisticRegression

# Logistic Regression Code Reference: https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
# Data source: https://www.kaggle.com/code/dyasin/week24ml-weather-dataset-rattle-package-weatheraus/data 

def handle_nulls(df):
    for col in df:
        col_median=df[col].median()
        df[col].fillna(col_median, inplace=True)

df = pd.read_csv("weatherAUS.csv")
y = pd.get_dummies(df.RainTomorrow, drop_first=True)
y = y.values.reshape(-1,1)

# Drop categorical columns
df.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'Evaporation', 'Sunshine', 'WindDir3pm', 'RainToday',  "RainTomorrow"],  axis=1, inplace=True)
handle_nulls(df)

# Normalize Data
df = (df-df.mean())/df.std()
X = df.values

# # Split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=44)

logregmodel = LogisticRegression(X_train, X_test, y_train, y_test)

weights, bias, losses = logregmodel.fit(lr=0.01, epochs=10, batchsize=10000)
y_pred = logregmodel.predict(weights, bias)

print("Accuracy: ",logregmodel.accuracy(y_pred))