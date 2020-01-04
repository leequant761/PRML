import itertools
import functools
import numpy as np


class PolynomialFeature(object):
    """
    polynomial features

    transforms input array with polynomial features

    Example
    =======
    x =
    [[a, b],
    [c, d]]

    y = PolynomialFeatures(degree=2).transform(x)
    y =
    [[1, a, b, a^2, a * b, b^2],
    [1, c, d, c^2, c * d, d^2]]
    """

    def __init__(self, degree=2):
        """
        construct polynomial features

        Parameters
        ----------
        degree : int
            degree of polynomial
        """
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        """
        transforms input array with polynomial features

        Parameters
        ----------
        x : (sample_size, p) ndarray
            input array

        Returns
        -------
        output : (sample_size, 1 + pC1 + ... + pCd) ndarray
            polynomial features
        """
        if x.ndim == 1:
            x = x[:, None] # N x 1
        x_t = x.transpose() # p x N
        features = [np.ones(len(x))] # N개의 1
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree): # items : 각 변수(item) 조합 (N 벡터 degree개) 튜플
                features.append(functools.reduce(lambda x, y: x * y, items)) # 곱하기 예를들어 x1 x3 x5가 선택됐으면 x1 * x3 * x5 벡터 나옴
        return np.asarray(features).transpose()

# 테스트 용
if __name__=='__main__':
    poly = PolynomialFeature(3)
    x = np.arange(100).reshape(25, 4) # 25 x 4
    poly.transform(x)