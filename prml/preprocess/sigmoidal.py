import numpy as np


class SigmoidalFeature(object):
    """
    Sigmoidal features

    1 / (1 + exp((m - x) @ c)
    """

    def __init__(self, mean, coef=1):
        """
        construct sigmoidal features

        Parameters
        ----------
        mean : (n_features, ndim) or (n_features,) ndarray
            center of sigmoid function
        coef : (ndim,) ndarray or int or float
            coefficient to be multplied with the distance
        """
        if mean.ndim == 1:
            mean = mean[:, None] # n_features x 1
        else:
            assert mean.ndim == 2 # 2차원 coordinate까지 지원
        if isinstance(coef, int) or isinstance(coef, float):
            if np.size(mean, 1) == 1: # mean n_column이 1이라면(1차원 coordinate)
                coef = np.array([coef])
            else: # 대신에 2차원인 경우 coef를 np.array로 넘겨줘야함
                raise ValueError("mismatch of dimension")
        else:
            assert coef.ndim == 1 # np.array로 들어왔을 때
            assert np.size(mean, 1) == len(coef)
        self.mean = mean
        self.coef = coef

    def _sigmoid(self, x, mean):
        return np.tanh((x - mean) @ self.coef * 0.5) * 0.5 + 0.5

    def transform(self, x):
        """
        transform input array with sigmoidal features

        Parameters
        ----------
        x : (sample_size, ndim) or (sample_size,) ndarray
            input array

        Returns
        -------
        output : (sample_size, n_features) ndarray
            sigmoidal features
        """
        if x.ndim == 1:
            x = x[:, None]
        else:
            assert x.ndim == 2
        assert np.size(x, 1) == np.size(self.mean, 1)
        basis = [np.ones(len(x))] # bias term
        for m in self.mean:
            basis.append(self._sigmoid(x, m))
        return np.asarray(basis).transpose() # sample_size X n_features

if __name__=='__main__':
    X_sigmoidal = SigmoidalFeature(np.linspace(-1, 1, 11), 10).transform(np.array([1]))
    X_sigmoidal2 = SigmoidalFeature(np.linspace(-1, 1, 12).reshape(-1, 2), np.array([10, 10]))#.transform(np.array([1]))
