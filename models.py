from relations import ACTIONS

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
import torch.nn as nn
import torch
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
        pred = self.clf.predict_proba(x)
        action = ACTIONS[self.clf.classes_[np.argmax(pred)]]
        alter_action = ACTIONS[self.clf.classes_[np.argsort(pred).squeeze()[-2]]]
        return action, alter_action


class SVM(Model):

    def __init__(self, *args, **kwargs):
        self.clf = LinearSVC(penalty='l1', dual=False, tol=1e-7)
    
    def train(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        pred = self.clf.decision_function(x)
        action = ACTIONS[self.clf.classes_[np.argmax(pred)]]
        alter_action = ACTIONS[self.clf.classes_[np.argsort(pred).squeeze()[-2]]]
        return action, alter_action


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
        pred = self.clf.predict_proba(x)
        action = ACTIONS[self.clf.classes_[np.argmax(pred)]]
        alter_action = ACTIONS[self.clf.classes_[np.argsort(pred).squeeze()[-2]]]
        return action, alter_action


class MultiLabel(Model):

    def __init__(self, *args, **kwargs):
        self.clf1 = BaggingClassifier(n_jobs=-1)
        self.clf2 = BaggingClassifier(n_jobs=-1)
        self.clf3 = SVC(kernel='rbf')
        self.actions = kwargs['actions']
    
    def train(self, x, y):
        y1 = np.array([self.actions[i].split('-')[0] for i in y])
        y2 = np.array([self.actions[i].split('-')[1] if self.actions[i] != 'SHIFT' else 'SHIFT' for i in y])
        y3 = np.array([self.actions[i].split('-')[2] if self.actions[i] != 'SHIFT' else 'SHIFT' for i in y])
        self.clf1.fit(x, y1)
        self.clf2.fit(x, y2)
        self.clf3.fit(x, y3)

    def predict(self, x):
        pred1 = self.clf1.predict_proba(x)
        pred2 = self.clf2.predict_proba(x)
        pred3 = self.clf3.predict_proba(x)
        a1 = 'REDUCE'
        # fix the action if needed
        a2 = self.clf2.classes_[np.argmax(pred2)] if self.clf2.classes_[np.argmax(pred2)] != 'SHIFT' else self.clf2.classes_[np.argsort(pred2).squeeze()[-2]] 
        a3 = self.clf3.classes_[np.argmax(pred3)] if self.clf3.classes_[np.argmax(pred3)] != 'SHIFT' else self.clf3.classes_[np.argsort(pred3).squeeze()[-2]]
        if self.clf1.classes_[np.argmax(pred1)] == 'SHIFT':
            action = 'SHIFT'
            alter_action = '-'.join([a1, a2, a3])
        else:
            action = '-'.join([a1, a2, a3])
            alter_action = 'INVALID'
        return action, alter_action


class Neural(Model):

    class Network(nn.Module):

        def __init__(self, n_features, hidden_size, num_classes):
            super().__init__(self)
            self.fc1 = nn.Linear(n_features, hidden_size)
            self.fc1.weight.data.fill_(1.0)
            self.fc2 = nn.Linear(hidden_size, num_classes)
            self.fc2.weight.data.fill_(1.0)

        def forward(self, x):
            x = nn.functional.relu(self.fc1(x))
            return nn.functional.relu(self.fc2(x))

    def __init__(self, *args, **kwargs):
        self.net = Neural.Network(kwargs['n_features'], hidden_size=128, num_classes=len(ACTIONS))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-4, momentum=0.9)
        self.num_iters = 200

    def train(self, x, y):
        for _ in range(self.num_iters):
            y_pred = self.net(torch.autograd.Variable(torch.tensor(x)))
            var = torch.autograd.Variable(torch.tensor(y))
            loss = self.criterion(y_pred, var)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        pred = self.net(torch.autograd.Variable(torch.tensor(x)))
        action = ACTIONS[pred.argmax()]
        _, indices = torch.sort(pred)
        alter_action = ACTIONS[indices[-2]]
        return action, alter_action
