import numpy as np
import pandas as pd

from utils import Utils

class LinearRegression():
  def __init__(self, X_train, X_test, y_train, y_test):
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    

  def h(self, weights):
    return Utils.mxmult(self.X_train, weights)


  # mean squared error
  def get_cost(self, weights):
    return ( Utils.mxmult(Utils.transpose(self.h(weights)-self.y_train),(self.h(weights)-self.y_train)) )/(2*self.y_train.shape[0])

    # n_rows =  (2*self.y_train.shape[0])
    # error = self.h(weights)-self.y_train
    # squared_error = Utils.mxmult(Utils.transpose(error), error)
    # mean_squared_error = squared_error / n_rows
    # return mean_squared_error


  def gradient_descent(self, weights, lr=0.1, epochs=10):
    
    costs = []
    n = self.X_train.shape[0]
    
    for _ in range(epochs):
      h_x = self.h(weights)
      cost_ = (1/n)*( Utils.mxmult(Utils.transpose(self.X_train), (h_x-self.y_train)) )
      weights = weights - (lr)*cost_
      costs.append(self.get_cost(weights))

      

    return weights, costs 

  
 
  def fit(self, lr=0.1, epochs=10):
    self.pad_1s_to_mx()
    weights = self.init_weights()

    # for _ in range(epochs):
    #   h_x = self.h(weights)


    weights, costs = self.gradient_descent(weights, lr, epochs)
    J = self.get_cost(weights)
    print("Cost: ", J)
    print("Parameters: ", weights)
    return J, weights

  def predict(self, weights):
    mu = np.mean(self.X_train[:,1:], axis=0)
    std = np.std(self.X_train[:,1:], axis=0)
        
    for i,x in enumerate(self.X_test):
      x_0 = (x[0] - mu[0])/std[0]
      x_1 = (x[1] - mu[1])/std[1]
      y = weights[0] + weights[1]*x_0 + weights[2]*x_1
    #   print("Predicted price of house: ", y)
    #   print("Actual price of house: ", self.y_test[i])


  def init_weights(self):
    return np.zeros((self.X_train.shape[1], 1))


  def pad_1s_to_mx(self):
    self.X_train = np.hstack((np.ones((self.X_train.shape[0], 1)), self.X_train))
    self.X_test = np.hstack((np.ones((self.X_test.shape[0], 1)), self.X_test))

if __name__ == "__main__":
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
    J, weights = linregmodel.fit(lr=0.01, epochs=400)

    # predict
    linregmodel.predict(weights)
