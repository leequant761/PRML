import numpy as np


class GaussianFeature(object):
    """
    Gaussian feature

    gaussian function = exp(-0.5 * (x - m) / v)
    """

    def __init__(self, mean, var):
        """
        construct gaussian features

        Parameters
        ----------
        mean : (n_features, ndim) or (n_features,) ndarray
            places to locate gaussian function at
        var : float
            variance of the gaussian function
        """
        if mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2 # 2차원 coordinate까지 지원
        assert isinstance(var, float) or isinstance(var, int)
        self.mean = mean
        self.var = var

    def _gauss(self, x, mean):
        return np.exp(-0.5 * np.sum(np.square(x - mean), axis=-1) / self.var) # np.sum은 모든 원소를 그냥 더하는 함수;

    def transform(self, x):
        """
        transform input array with gaussian features

        Parameters
        ----------
        x : (sample_size, ndim) or (sample_size,)
            input array

        Returns
        -------
        output : (sample_size, n_features)
            gaussian features
        """
        if x.ndim == 1:
            x = x[:, None]
        else:
            assert x.ndim == 2
        assert np.size(x, 1) == np.size(self.mean, 1)
        basis = [np.ones(len(x))] # bias term
        for m in self.mean: # 각 평균 지점 마다
            basis.append(self._gauss(x, m)) # 들어온 x를 가우시안 베이시스의 값으로 바꾸고 list에 추가
        return np.asarray(basis).transpose() # sample_size X n_features

# 테스트용
if __name__=='__main__':
    X_G = GaussianFeature(np.linspace(-1, 1, 11), 0.1)
    X_G.transform(np.array([1]))
    X_G2 = GaussianFeature(np.linspace(-1, 1, 12).reshape(-1, 2), 0.1)
    X_G2.transform(np.array([[1, 2]]))
    