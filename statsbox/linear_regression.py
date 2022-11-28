import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from utils import Utils

class LinearRegression():
  def __init__(self):
    self.cost = 0
    self.weights = []
    self.bias = 0
    

  def h(self, X):
    return np.dot(X, self.weights) 


  # mean squared error
  def get_cost(self,X, y):
    n_rows =  (2*y.shape[0])
    error = self.h(X)-y
    squared_error = np.dot(error.T, error)
    mean_squared_error = squared_error / n_rows
    return mean_squared_error


  def gradient_descent(self, X, y, useAdagrad, lr=0.1, epochs=10):
    costs = []
    n = X.shape[0]
    if useAdagrad:
      adagradient = np.zeros((X.shape[1], 1))
    for _ in range(epochs):
      
      h_x = self.h(X)
      cost_ = (1/n)*(np.dot(X.T, (h_x-y)) )
      if(not useAdagrad):
        self.weights = self.weights - lr*cost_   
      else:
        step, adagradient = Utils.adagrad(lr, cost_, adagradient)
        self.weights = self.weights - step
      costs.append(self.get_cost(X, y))

    return costs 
 
  def fit(self, X, y, useAdagrad=False, lr=0.1, epochs=10):
    X = self.pad_1s_to_mx(X)
    self.weights = self.init_weights(X)
    costs = self.gradient_descent(X, y, useAdagrad, lr, epochs)
    self.cost = self.get_cost(X, y)
    
  def mse(self, y, y_hat):
    return mean_squared_error(y, y_hat)


  def predict(self, X):
    X = self.pad_1s_to_mx(X)
    return self.h(X)


  def init_weights(self, X):
    return np.zeros((X.shape[1], 1))


  def pad_1s_to_mx(self, X):
    return np.hstack((np.ones((X.shape[0], 1)), X))
    
