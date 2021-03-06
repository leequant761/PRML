import numpy as np
from prml.linear.regression import Regression


class RidgeRegression(Regression):
    """
    Ridge regression model

    w* = argmin |t - X @ w| + alpha * |w|_2^2
    """

    def __init__(self, alpha:float=1.):
        self.alpha = alpha

    def fit(self, X:np.ndarray, t:np.ndarray):
        """
        maximum a posteriori estimation of parameter

        Parameters
        ----------
        X : (N, D) np.ndarray
            training data independent variable
        t : (N,) np.ndarray
            training data dependent variable
        """

        eye = np.eye(np.size(X, 1)) # N x N Identity matrix
        self.w = np.linalg.solve(self.alpha * eye + X.T @ X, X.T @ t) # 선형시스템을 풀어라; 리지 클로즈드 폼 inv(X'X + aI) @ X'y = beta 를 이용해서

    def predict(self, X:np.ndarray):
        """
        make prediction given input

        Parameters
        ----------
        X : (N, D) np.ndarray
            samples to predict their output

        Returns
        -------
        (N,) np.ndarray
            prediction of each input
        """
        return X @ self.w
