from collections import Counter
from abc import ABC, abstractmethod
from features import ACTIONS
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np


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
        pred = self.clf.decision_function(x)
        action = ACTIONS[self.clf.classes_[np.argmax(pred)]]
        alter_action = ACTIONS[self.clf.classes_[np.argsort(pred).squeeze()[-2]]]
        return action, alter_action


class SVM(Model):

    def __init__(self, *args, **kwargs):
        self.clf = BaggingClassifier(LinearSVC(penalty='l1', dual=False, tol=1e-7), n_jobs=-1)

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
        self.clf = RandomForestClassifier(n_estimators=n_estimators)

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
        self.clf3 = BaggingClassifier(n_jobs=-1)

    def train(self, x, y):
        y1 = np.array([ACTIONS[i].split('-')[0] for i in y])
        y2 = np.array([ACTIONS[i].split('-')[1] if ACTIONS[i] != 'SHIFT' else 'SHIFT' for i in y])
        y3 = np.array([ACTIONS[i].split('-')[2] if ACTIONS[i] != 'SHIFT' else 'SHIFT' for i in y])
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
            super().__init__()
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print('GPU is available')
            else:
                self.device = torch.device('cpu')
                print('GPU not available, CPU used')
            self.fc1 = nn.Linear(n_features, hidden_size)
            self.fc1.weight.data.fill_(1.0)
            self.fc2 = nn.Linear(hidden_size, num_classes)
            self.fc2.weight.data.fill_(1.0)

        def forward(self, x):
            x = nn.functional.relu(self.fc1(x.float()))
            return nn.functional.relu(self.fc2(x.float()))

    def __init__(self, *args, **kwargs):
        self.net = Neural.Network(n_features=kwargs['n_features'],
                                  hidden_size=128,
                                  num_classes=len(ACTIONS))
        self.net.to(self.net.device)                                  
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.n_epochs = 50

    def train(self, x, y):

        train = TensorDataset(torch.tensor(x), torch.tensor(y))
        for epoch in range(self.n_epochs):
            trainloader = DataLoader(train, batch_size=32, shuffle=True, num_workers=2)

            for _, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs.to(self.net.device)), Variable(labels.to(self.net.device))
                self.optimizer.zero_grad()
                y_pred = self.net(Variable(torch.tensor(inputs).to(self.net.device)))
                loss = self.criterion(y_pred, Variable(torch.tensor(labels).to(self.net.device)))
                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch: {epoch + 1}/{self.n_epochs}.............', end=' ')
                print(f'Loss: {loss.item():.4f}')

    def predict(self, x):
        pred = self.net(Variable(torch.tensor(x.squeeze()).to(self.net.device)))
        action = ACTIONS[pred.argmax()]
        _, indices = torch.sort(pred)
        alter_action = ACTIONS[indices[-2]]
        return action, alter_action


class RNN(Model):

    class Network(nn.Module):

        def __init__(self, input_size, output_size, hidden_dim, n_layers, unique_labels, max_seq_len):
            super().__init__()
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print('GPU is available')
            else:
                self.device = torch.device('cpu')
                print('GPU not available, CPU used')

            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
            self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_size)
            self.input_size = input_size
            self.output_size = output_size
            self.unique_labels = unique_labels
            self.max_seq_len = max_seq_len

        def forward(self, x):
            batch_size = x.size(0)
            hidden = self.init_hidden(batch_size)
            out, hidden = self.rnn(x, hidden)
            out = out.contiguous().view(-1, self.hidden_dim)
            out = self.fc(out)
            return out, hidden

        def init_hidden(self, batch_size):
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
            return hidden

    def __init__(self, *args, **kwargs):
        self.unique_labels = np.unique([sample.action for sample in kwargs['samples']])
        self.max_seq_len = max(len(tree._samples) for tree in kwargs['trees'])
        self.input_size = kwargs['n_features']
        self.output_size = len(self.unique_labels)
        self.net = RNN.Network(input_size=kwargs['n_features'], output_size=self.output_size,
                               hidden_dim=256, n_layers=2, unique_labels=self.unique_labels,
                               max_seq_len=self.max_seq_len)
        self.net.to(self.net.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)

        self.sents_idx = kwargs['sents_idx']

    def train(self, x, y):
        batch_size = self.sents_idx.count('')
        input_seq = np.zeros((batch_size, self.max_seq_len, self.input_size), dtype=np.float32)
        target_seq = np.zeros((batch_size, self.max_seq_len, self.output_size), dtype=np.float32)
        old_idx = 0
        j = -1
        for idx in range(len(y)):
            if self.sents_idx[idx] == '':
                j += 1
                input_seq[j] = self._add_padding(x[old_idx:idx], shape=(self.max_seq_len, self.input_size))
                target_seq[j] = self._add_padding(y[old_idx:idx], shape=(self.max_seq_len, self.output_size), one_hot=True, labels=self.unique_labels)
                old_idx = idx
        input_seq = torch.from_numpy(input_seq)
        target_seq = torch.Tensor(target_seq)
        n_epochs = 100
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            input_seq = input_seq.to(self.net.device)
            output, hidden = self.net(input_seq)
            loss = self.criterion(output, np.argmax(target_seq, axis=2).view(-1).long().to(self.net.device))
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch: {epoch + 1}/{n_epochs}.............', end=' ')
                print(f'Loss: {loss.item():.4f}')

    def _add_padding(self, x, shape, one_hot=False, labels=None):
        arr = np.zeros(shape=shape)
        x_len = len(x)
        for i in range(shape[0]):
            if i < x_len:
                if one_hot:
                    arr[i] = self._one_hot_encode(x[i], labels)
                else:
                    arr[i] = x[i]
        return arr

    def _one_hot_encode(self, label, labels):
        vec = np.zeros(len(labels), dtype=np.float32)
        vec[np.where(labels == ACTIONS[label])] = 1.0
        return vec

    def predict(self, x):
        pred, hidden = self.net(Variable(torch.tensor(x).to(self.net.device)))
        prob = nn.functional.softmax(pred, dim=0).data
        actions_idx = torch.sort(prob, descending=True, dim=-1)[1][:,0]
        alter_idx = torch.sort(prob, descending=True, dim=-1)[1][:,1]
        actions = [self.net.unique_labels[i] for i in actions_idx]
        alter_actions = [self.net.unique_labels[i] for i in alter_idx]

        return actions, alter_actions


class VoteModel(Model):

    def __init__(self, *args, **kwargs):
        self.models = [model(*args, **kwargs) for model in kwargs['models']]

    def train(self, x, y):
        for model in self.models:
            model.train(x, y)

    def predict(self, x):
        actions_freq = Counter(action
                               for model in self.models
                               for action in model.predict(x))
        most_common_action = actions_freq.most_common(1)[0][0]
        second_most_common_action = actions_freq.most_common(2)[1][0]
        return most_common_action, second_most_common_action
