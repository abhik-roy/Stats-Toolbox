import numpy as np
class Utils():
    def gradient_descent(X, y, hypothesis):
        dw = (1/X.shape[0])*np.dot(X.T, (hypothesis - y))
        db = (1/X.shape[0])*np.sum((hypothesis - y)) 
        return dw, db


    def transpose(mx):
        result = [[0 for i in range(mx.shape[0])] for j in range(mx.shape[1])]
        for i in range(len(mx)):
            for j in range(len(mx[0])):
                result[j][i] = mx[i][j]
        return np.array(result)


    def mxmult(mx1, mx2):
        result = [[0 for i in range(0, mx2.shape[1])] for j in range(0, mx1.shape[0])]
    
        for i in range(len(mx1)): #rows of mx1
            for j in range(len(mx2[0])): #cols of mx2
                for k in range(len(mx2)): # rows of mx2
                    result[i][j] += mx1[i][k] * mx2[k][j]
        return np.array(result)

    