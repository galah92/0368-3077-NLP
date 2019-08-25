from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def train(self, x_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, x_train, y_train):
        pass
