import numpy as np
import pandas as pd

from utils import Utils

class LogisticRegression():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def gradients(self, X, y, y_hat):
    
        n = X.shape[0]

        dw = (1/n)*Utils.mxmult(Utils.transpose(self.X_train), (y_hat - y))
        db = (1/n)*np.sum((y_hat - y)) 
        
        return dw, db


    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))

    def h(self, x, beta, bias):
        print(x.shape, beta.shape, "eee", Utils.mxmult(x, beta).shape)
        return self.sigmoid(Utils.mxmult(x, beta) + bias)

    def cost(self, weights):
        n_rows =  (2*self.y_train.shape[0])
        h_x = self.h(weights)
        cost = -(self.y_train*(np.log(h_x)) - (1-self.y_train)*np.log(1-h_x))/n_rows
        return cost

    def accuracy(y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y)
        return accuracy

    def fit(self, lr=0.1, epochs=10, batchsize=20):
        n = self.X_train.shape[0]
        w = np.zeros((self.X_train.shape[1], 1))
        b =  np.zeros((batchsize, 1))

        for epoch in range(0,epochs):
            for i in range((n-1)//batchsize + 1):
                first_idx = i*batchsize
                last_idx = first_idx + batchsize
                xb, yb = self.X_train[first_idx:last_idx], self.y_train[first_idx:last_idx]
                y_hat = self.h(xb, w, b)
                dw, db = self.gradients(xb, yb, y_hat)

                w = w - lr*dw
                b = b - lr*dw
            
            l = get_cost(self.y_train, self.h(self.X_train, w, b))
            losses.append(l)

        print("Losses: ", losses)
        return w, b, losses


def handle_nulls(df):
    for col in df:
        col_median=df[col].median()
        df[col].fillna(col_median, inplace=True)
    
if __name__ == "__main__":
    # Logistic Regression Code Reference: https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
    # Data source: https://www.kaggle.com/code/dyasin/week24ml-weather-dataset-rattle-package-weatheraus/data 
    
    df = pd.read_csv("weatherAUS.csv")
    y = pd.get_dummies(df.RainTomorrow, drop_first=True)
    y = y.values.reshape(-1,1)
    # df = encode_categorical_vars(df)
    df.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday',  "RainTomorrow"],  axis=1, inplace=True)
    handle_nulls(df)

    # Normalize Data
    df = (df-df.mean())/df.std()
    X = df.values

    # # Split into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=44)
   
    logregmodel = LogisticRegression(X_train, X_test, y_train, y_test)

    J, beta = logregmodel.fit(lr=0.01, epochs=400)
    # linregmodel.predict(beta)