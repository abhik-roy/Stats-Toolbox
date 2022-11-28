import numpy as np
import pandas as pd
from utils import Utils


class LogisticRegression():
    def __init__(self):
        self.weights = []
        self.bias = 0

    def gradients(self, X, y, y_hat):
        n = X.shape[0]
        dw = (1/n)*np.dot(X.T, (y_hat - y))
        db = (1/n)*np.sum((y_hat - y))
        return dw, db

    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))

    def h(self, x):
        return self.sigmoid(np.dot(x, self.weights) + self.bias)

    def get_cost(self, X, y):
        n_rows = y.shape[0]
        h_x = self.h(X)
        cost = -(y*(np.log(h_x)) - (1-y)*np.log(1-h_x))/n_rows
        return cost

    def accuracy(y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y)
        return accuracy

    def fit(self, X, y, useAdagrad=False, lr=0.1, epochs=10, batchsize=100):
        
        if not isinstance(X, np.ndarray):
            print(X.type())
            print("incorrect input for X_train, must be a list")
            return
        if not isinstance(y, np.ndarray):
            print("incorrect input for y_train, must be a list")
            return

        n = X.shape[0]
        if useAdagrad:
            adagradient = np.zeros((X.shape[1], 1))
        self.weights = np.zeros((X.shape[1], 1))
        losses = []

        for epoch in range(0, epochs):
            for i in range((n-1)//batchsize + 1):
                first_idx = i*batchsize
                last_idx = first_idx + batchsize
                xb, yb = X[first_idx:last_idx], y[first_idx:last_idx]
                y_hat = self.h(xb)
                dw, db = self.gradients(xb, yb, y_hat)
                if(not useAdagrad):
                    self.weights = self.weights - lr*dw   
                else:
                    step, adagradient = Utils.adagrad(lr, dw, adagradient)
                    self.weights = self.weights - step
                self.bias = self.bias - lr*db

            l = self.get_cost(X, y)
            losses.append(l)

    def predict(self, X):
        
        if not isinstance(X, np.ndarray):
            print(X.type())
            print("incorrect input for X_test, must be a list")
            return
        

        preds = self.h(X)
        pred_class = [1 if i > 0.5 else 0 for i in preds]
        return np.array(pred_class)

    def accuracy(self, actual, predicted):
        predicted = predicted.reshape(actual.shape[0],1)
        accuracy = np.sum(actual == predicted) / len(actual)
        return accuracy
