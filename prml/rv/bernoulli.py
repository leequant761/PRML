import numpy as np
from prml.rv.rv import RandomVariable
from prml.rv.beta import Beta


class Bernoulli(RandomVariable):
    """
    Bernoulli distribution
    p(x|mu) = mu^x (1 - mu)^(1 - x)
    """

    def __init__(self, mu=None):
        """
        construct Bernoulli distribution

        Parameters
        ----------
        mu : np.ndarray or Beta
            probability of value 1 for each element
        """
        super().__init__() # self.parameter = {} 라고 정의
        self.mu = mu # 프로퍼티

    @property
    def mu(self):
        return self.parameter["mu"] # mu는 self.parameter에 담김

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, (int, float, np.number)): # 덕타이핑
            if mu > 1 or mu < 0:
                raise ValueError(f"mu must be in [0, 1], not {mu}")
            self.parameter["mu"] = np.asarray(mu)
        elif isinstance(mu, np.ndarray): # mu에 np.array로 들어오면
            if (mu > 1).any() or (mu < 0).any(): 
                raise ValueError("mu must be in [0, 1]")
            self.parameter["mu"] = mu
        elif isinstance(mu, Beta): # mu에 베타클래스의 객체로 들어오면
            self.parameter["mu"] = mu
        else:
            if mu is not None: # 그 외엔 지원 안함
                raise TypeError(f"{type(mu)} is not supported for mu")
            self.parameter["mu"] = None # mu에 None이 들어오면

    @property
    def ndim(self):
        if hasattr(self.mu, "ndim"): # mu에 np.array로 들어오면
            return self.mu.ndim
        else:
            return None

    @property
    def size(self):
        if hasattr(self.mu, "size"): # mu에 np.array로 들어오면
            return self.mu.size
        else:
            return None

    @property
    def shape(self):
        if hasattr(self.mu, "shape"): # mu에 np.array로 들어오면
            return self.mu.shape
        else:
            return None

    def _fit(self, X): # 부모클래스인 rv의 fit 메서드(deprecated)를 쓸 때 사용될 예정; 이렇게 짠 이유는 모든 rv마다 docstring을 다는 것은 낭비기 떄문에
        if isinstance(self.mu, Beta): # prior가 Beta분포 객체면
            self._bayes(X) # posterior로 Beta분포 객체 반환
        elif isinstance(self.mu, RandomVariable): # 그 외(non-conjugate)의 분포는 안받아줌
            raise NotImplementedError
        else:
            self._ml(X) # Frequentist의 maximum likelihood estimator 반환 

    def _ml(self, X): # rv maximum likelihood
        n_zeros = np.count_nonzero((X == 0).astype(np.int)) # (p, )
        n_ones = np.count_nonzero((X == 1).astype(np.int)) # (p, )
        assert X.size == n_zeros + n_ones, (
            "{X.size} is not equal to {n_zeros} plus {n_ones}"
        )
        self.mu = np.mean(X, axis=0) # 평균이 곧 mle

    def _map(self, X): # 부모클래스인 rv의 map 메서드(deprecated)를 쓸 때 사용될 예정;
        assert isinstance(self.mu, Beta)
        assert X.shape[1:] == self.mu.shape
        n_ones = (X == 1).sum(axis=0)
        n_zeros = (X == 0).sum(axis=0)
        assert X.size == n_zeros.sum() + n_ones.sum(), (
            f"{X.size} is not equal to {n_zeros} plus {n_ones}"
        )
        n_ones = n_ones + self.mu.n_ones
        n_zeros = n_zeros + self.mu.n_zeros
        self.prob = (n_ones - 1) / (n_ones + n_zeros - 2)

    def _bayes(self, X):
        assert isinstance(self.mu, Beta) # mu에는 사전분포로 베타분포 클래스 객체
        assert X.shape[1:] == self.mu.shape # mu가 p개면 데이터도 n x p 여야함
        n_ones = (X == 1).sum(axis=0) # 1의 개수 ; (p, )
        n_zeros = (X == 0).sum(axis=0) # 0의 개수 ; (p, )
        assert X.size == n_zeros.sum() + n_ones.sum(), (
            "input X must only has 0 or 1"
        )
        self.mu.n_zeros += n_zeros # 베타분포 객체의 n_zeros attribute를 업데이트 (posterior)
        self.mu.n_ones += n_ones # 베타분포 객체의 n_ones attribute를 업데이트 (posterior)

    def _pdf(self, X): # 부모클래스인 rv의 pdf 메서드를 쓸 때 사용될 예정;
        assert isinstance(mu, np.ndarray)
        return np.prod(
            self.mu ** X * (1 - self.mu) ** (1 - X)
        )

    def _draw(self, sample_size=1): # 부모클래스인 rv의 draw 메서드를 쓸 때 사용될 예정;
        if isinstance(self.mu, np.ndarray):
            return (
                self.mu > np.random.uniform(size=(sample_size,) + self.shape)
            ).astype(np.int)
        elif isinstance(self.mu, Beta): # 베타 prior일 떄 predictive distribution은 p(1) = a / (a + b)
            return (
                self.mu.n_ones / (self.mu.n_ones + self.mu.n_zeros)
                > np.random.uniform(size=(sample_size,) + self.shape) # samplesize x p
            ).astype(np.int)
        elif isinstance(self.mu, RandomVariable): # fit은 안해주지만 뽑아줄 순 있음 # 확률변수일 때 뽑는 원리는 확률변수에서 먼저 뽑고 그게 주어진 상황에서 유니폼 분포에서 하나 뽑아서 그게 먼저 뽑은것보다 크냐 적냐로
            return (
                self.mu.draw(sample_size)
                > np.random.uniform(size=(sample_size,) + self.shape)
            )
