from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
import numpy as np

from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def train(self, x, y):
        pass
    
    @abstractmethod
    def predict(self, x):
        pass


class SGD(Model):

    def __init__(self, *args, **kwargs):
        self.clf = OneVsRestClassifier(SGDClassifier(alpha=0.1, penalty='l2', n_jobs=-1))

    def train(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)


class SVM(Model):

    def __init__(self, *args, **kwargs):
        self.clf = LinearSVC(penalty='l1', dual=False, tol=1e-7)
    
    def train(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)


class RandomForest(Model):

    def __init__(self, *args, **kwargs):
        # TODO: [Eyal] add SelectFromModel for feature reduction.
        n_estimators = 100
        self.clf = BaggingClassifier(RandomForestClassifier(n_estimators=n_estimators),
                                     max_samples=1.0 / n_estimators,
                                     n_estimators=n_estimators,
                                     n_jobs=-1)
    
    def train(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)


class MultiLabel(Model):

    def __init__(self, *args, **kwargs):
        self.clf1 = BaggingClassifier(n_jobs=-1)
        self.clf2 = BaggingClassifier(n_jobs=-1)
        self.clf3 = SVC(kernel='rbf')
        self.actions = kwargs['actions']
    
    def train(self, x, y):
        y1 = np.array([self.actions[i].split('-')[0] for i in y])
        y2 = np.array([self.actions[i].split('-')[1] if self.actions[i] != 'SHIFT' else 'SHIFT' for i in y])
        y_3 = np.array([self.actions[i].split('-')[2] if self.actions[i] != 'SHIFT' else 'SHIFT' for i in y])
        self.clf1.fit(x, y1)
        self.clf2.fit(x, y2)
        self.clf3.fit(x, y_3)

    def predict(self, x):
        return NotImplementedError()
