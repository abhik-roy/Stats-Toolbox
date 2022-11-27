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
        # print("aa", x[0])
        # print((Utils.mxmult(x, beta)))
        return self.sigmoid(Utils.mxmult(x, beta) + bias)

    def get_cost(self, weights, bias):
        n_rows =  (self.y_train.shape[0])
        # print(self.X_train.shape, weights.shape, bias,"aeee")
        h_x = self.h(self.X_train, weights, bias)
        # print("hx", (self.X_train))
        cost = -(self.y_train*(np.log(h_x)) - (1-self.y_train)*np.log(1-h_x))/n_rows
        return cost

    def accuracy(y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y)
        return accuracy

    def fit(self, lr=0.1, epochs=10, batchsize=100):
        n = self.X_train.shape[0]
        w = np.zeros((self.X_train.shape[1], 1))
       
        b =  0
        losses = []

        for epoch in range(0,epochs):
            for i in range((n-1)//batchsize + 1):
                first_idx = i*batchsize
                last_idx = first_idx + batchsize
                xb, yb = self.X_train[first_idx:last_idx], self.y_train[first_idx:last_idx]
                y_hat = self.h(xb, w, b)
                dw, db = self.gradients(xb, yb, y_hat) # yhat is nan
                w = w - lr*dw   
                b = b - lr*db
                
            
            l = self.get_cost(w, b)
            losses.append(l)
            break

        # print("Losses: ", losses)
        return w, b, losses
    
    def predict(self, w, b):
        X = self.X_test
        preds = self.h(X, w, b)
        
        # Empty List to store predictions.
        pred_class = []
        # if y_hat >= 0.5 --> round up to 1
        # if y_hat < 0.5 --> round up to 1
        pred_class = [1 if i > 0.5 else 0 for i in preds]
        
        return np.array(pred_class)

    def accuracy(self, predicted):
        accuracy = np.sum(self.y_test[0] == predicted) / len(self.y_test)
        return accuracy


def handle_nulls(df):
    for col in df:
        col_median=df[col].median()
        df[col].fillna(col_median, inplace=True)
    
if __name__ == "__main__":
    # Logistic Regression Code Reference: https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
    # Data source: https://www.kaggle.com/code/dyasin/week24ml-weather-dataset-rattle-package-weatheraus/data 
    
    df = pd.read_csv("weatherAUS.csv")
    df = df.iloc[:10]
    y = pd.get_dummies(df.RainTomorrow, drop_first=True)
    y = y.values.reshape(-1,1)
    # df = encode_categorical_vars(df)
    df.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'Evaporation', 'Sunshine', 'WindDir3pm', 'RainToday',  "RainTomorrow"],  axis=1, inplace=True)
    handle_nulls(df)

    # Normalize Data
    df = (df-df.mean())/df.std()
    X = df.values

    # # Split into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=44)
   
    logregmodel = LogisticRegression(X_train, X_test, y_train, y_test)

    weights, bias, losses = logregmodel.fit(lr=0.01, epochs=10)
    y_pred = logregmodel.predict(weights, bias)
    
    print("Accuracy: ",logregmodel.accuracy(y_pred))
