import numpy as np
import pandas as pd
from statsbox.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as sklearn_lin
from sklearn.metrics import mean_squared_error

def handle_nulls(df):
    for col in df:
        col_median=df[col].median()
        df[col].fillna(col_median, inplace=True)

def run_sklearn_model():
    sklearn_model = sklearn_lin()
    sklearn_model.fit(X_train, y_train)
    y_pred_test = sklearn_model.predict(X_test)
    print('Sklearn mean squared error:', mean_squared_error(y_test, y_pred_test))
    # print('Sklearn model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

# Linear Regression Code Reference: https://towardsdatascience.com/coding-linear-regression-from-scratch-c42ec079902 
# Data source: https://github.com/kumudlakara/Medium-codes/blob/main/linear_regression/house_price_data.txt
df = pd.read_csv("house_price_data.txt", index_col=False)
df.columns = ["housesize", "rooms", "price"]
handle_nulls(df)

# Normalize the data
df = (df-df.mean())/df.std()

# store non-label values into matrix X
X = df.iloc[:, :-1].values

# store labels into y
y = df.iloc[:, -1].values.reshape(-1,1)

# split dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=44)

# initialize model
linregmodel = LinearRegression()
# fit
linregmodel.fit(X_train, y_train, lr=0.1, epochs=400)
# predict
y_pred = linregmodel.predict( X_test)
print("Model mean squared error: ", linregmodel.mse(y_test, y_pred))


linregmodel_ada = LinearRegression()
linregmodel_ada.fit(X_train, y_train, useAdagrad=True, lr=0.1, epochs=400)
y_pred = linregmodel_ada.predict(X_test)
print("Model mean squared error (with Adagrad): ",linregmodel.mse(y_test, y_pred))


run_sklearn_model()