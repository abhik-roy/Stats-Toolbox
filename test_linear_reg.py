import numpy as np
import pandas as pd
from statsbox.linear_regression import LinearRegression

# Linear Regression Code Reference: https://towardsdatascience.com/coding-linear-regression-from-scratch-c42ec079902 
# Data source: https://github.com/kumudlakara/Medium-codes/blob/main/linear_regression/house_price_data.txt
df = pd.read_csv("house_price_data.txt", index_col=False)
df.columns = ["housesize", "rooms", "price"]

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
linregmodel = LinearRegression(X_train, X_test, y_train, y_test)

# fit
linregmodel.fit(lr=0.01, epochs=400)

# predict
err = linregmodel.predict()
print("Model Error: ", err)