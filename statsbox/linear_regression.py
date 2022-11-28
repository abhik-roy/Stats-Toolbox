import numpy as np
import pandas as pd


class LinearRegression():
  def __init__(self, X_train, X_test, y_train, y_test):
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.pad_1s_to_mx()

    self.cost = 0
    self.weights = self.init_weights()
    

  def h(self):
    return np.dot(self.X_train, self.weights)


  # mean squared error
  def get_cost(self):
    n_rows =  (2*self.y_train.shape[0])
    error = self.h()-self.y_train
    squared_error = np.dot(error.T, error)
    mean_squared_error = squared_error / n_rows
    return mean_squared_error


  def gradient_descent(self, lr=0.1, epochs=10):
    
    costs = []
    n = self.X_train.shape[0]
    
    for _ in range(epochs):
      h_x = self.h()
      cost_ = (1/n)*(np.dot(self.X_train.T, (h_x-self.y_train)) )
      self.weights = self.weights - (lr)*cost_
      costs.append(self.get_cost())

    return costs 
 
  def fit(self, lr=0.1, epochs=10):
    costs = self.gradient_descent(lr, epochs)
    self.cost = self.get_cost()
    

  def predict(self):
    mu = np.mean(self.X_train[:,1:], axis=0)
    std = np.std(self.X_train[:,1:], axis=0)
        
    for i,x in enumerate(self.X_test):
      x_0 = (x[0] - mu[0])/std[0]
      x_1 = (x[1] - mu[1])/std[1]
      y = self.weights[0] + self.weights[1]*x_0 + self.weights[2]*x_1
      print("Predicted price of house: ", y)
      print("Actual price of house: ", self.y_test[i])


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
