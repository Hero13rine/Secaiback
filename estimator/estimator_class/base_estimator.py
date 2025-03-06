from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    """所有估计器的统一抽象接口"""
    @abstractmethod
    def predict(self, x):
        pass

    # @abstractmethod
    # def fit(self, x, y):
    #     pass