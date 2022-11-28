import numpy as np
class Utils():
    def adagrad(lr, dw, adagradient):
        adagradient += dw**2
        step = lr / (np.sqrt(adagradient + 1e-6)) * dw
        return step, adagradient