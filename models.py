from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.multiclass import OneVsRestClassifier
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from relations_inventory import ind_toaction_map
from features import extract_features


hidden_size = 128
lr = 1e-4  # learning rate

def print_model(clf, samples_num, labels_num):
    print("Samples = {}, Labels = {}".format(samples_num, labels_num))
    print(clf)

class Network(nn.Module):
    def __init__(self, n_features, hidden_size, num_classes):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.fc1.weight.data.fill_(1.0)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fc2.weight.data.fill_(1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


def svm_model(trees, samples, labels, vocab, tag_to_ind_map, n_jobs, verbose=0):
    n_estimators = 10
    clf = LinearSVC(penalty='l1', dual=False, tol=1e-7, verbose=verbose)
    print_model(clf, len(samples), len(labels))
    X, y = extract_features(trees, samples, vocab, None, tag_to_ind_map)
    clf.fit(X, y)
    return clf


def random_forest_model(trees, samples, labels, vocab, tag_to_ind_map, n_jobs, verbose=0):
    n_estimators = 10
    clf = BaggingClassifier(RandomForestClassifier(n_estimators = 100, verbose=verbose),
        max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=n_jobs)
    # TODO : Eyal, add SelectFromModel for feature reduction.
    print_model(clf, len(samples), len(labels))
    X, y = extract_features(trees, samples, vocab, None, tag_to_ind_map)
    clf.fit(X, y)
    return clf


def neural_network_model(trees, samples, vocab, tag_to_ind_map, iterations=200, subset_size=5000):
    
    num_classes = len(ind_toaction_map)

    [x_vecs, _] = extract_features(trees, samples, vocab, 1, tag_to_ind_map)

    print("num features {}, num classes {}".format(len(x_vecs[0]), num_classes))
    print("Running neural model")

    net = Network(len(x_vecs[0]), hidden_size, num_classes)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for i in range(iterations):
        [x_vecs, y_labels] = extract_features(trees, samples, vocab, subset_size, tag_to_ind_map)
        y_pred = net(Variable(torch.tensor(x_vecs, dtype=torch.float)))
        loss = criterion(y_pred, Variable(torch.tensor(y_labels, dtype=torch.long)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return net


def neural_net_predict(net, x_vecs):
    return net(Variable(torch.tensor(x_vecs, dtype=torch.float)))


def sgd_model(trees, samples, labels, vocab, tag_to_ind_map, n_jobs, iterations=200, subset_size=500, verbose=0):
    n_estimators = 10
    clf = linear_model.SGDClassifier(penalty='l1', verbose=verbose, n_jobs=n_jobs)
    print_model(clf, len(samples), len(labels))
    X, y = extract_features(trees, samples, vocab, None, tag_to_ind_map)
    clf.fit(X, y)
    # for _ in range(iterations):
    #     [x_vecs, y_labels] = extract_features(trees, samples, vocab, subset_size, tag_to_ind_map)
    #     linear_train(clf, x_vecs, y_labels, classes)
    #     classes = None
    return clf