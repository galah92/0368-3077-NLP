from sklearn import linear_model
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from relations_inventory import ind_toaction_map
from features import extract_features
import numpy as np


hidden_size = 128
lr = 1e-4  # learning rate

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


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


def add_padding(X, shape):
    arr = np.zeros(shape=shape)
    max_len = shape[0]
    x_len = len(X)

    for i in range(max_len):
        if i < x_len:
            arr[i] = X[i]

    return arr


def rnn_model(trees, samples, vocab, tag_to_ind_map):
    unique_labels = np.unique([sample.action for sample in samples])
    output_size = len(unique_labels)
    input_size = len(extract_features(trees, samples, vocab, 1, tag_to_ind_map)[0][0])
    model = RNN_Model(input_size=input_size, output_size=output_size, hidden_dim=256, n_layers=2)
    model.to(device)

    # RNN hyperparameters
    n_epochs = 100
    lr=0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train data
    x_vecs, y_labels, sents_idx = extract_features(trees, samples, vocab, None, tag_to_ind_map, rnn=True)
    max_seq_len = max(tree._root.span[1] for tree in trees)
    batch_size = sents_idx.count("")
    input_seq = np.zeros((batch_size, max_seq_len, input_size), dtype=np.float32)
    target_seq = np.zeros((batch_size), dtype=np.float32)

    old_idx = 0
    for idx in range(len(y_labels)):
        if sents_idx[idx] == "":
            target_seq[idx] = y_labels[idx]
            old_idx = idx

    input_seq = torch.from_numpy(input_seq[np.newaxis,:])
    target_seq = torch.Tensor(target_seq[np.newaxis,:])

    # Training
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        input_seq = input_seq.to(device)
        output, hidden = model(input_seq)
        loss = criterion(output, target_seq.view(-1).long())
        loss.backward() 
        optimizer.step() 
        
        if epoch%10 == 0:
            print (f'Epoch: {epoch}/{n_epochs}.............', end=' ')
            print (f'Loss: {loss.item():.4f}')


class RNN_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN_Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden


def svm_model(trees, samples, vocab, tag_to_ind_map, n_jobs, verbose=0):
    clf = LinearSVC(penalty='l1', dual=False, tol=1e-7, verbose=verbose)

    X, y = extract_features(trees, samples, vocab, None, tag_to_ind_map)
    clf.fit(X, y)
    return clf


def random_forest_model(trees, samples, vocab, tag_to_ind_map, n_jobs, verbose=0):
    n_estimators = 10
    clf = BaggingClassifier(RandomForestClassifier(n_estimators = 100, verbose=verbose),
        max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=n_jobs)
    # TODO : Eyal, add SelectFromModel for feature reduction.
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


def sgd_model(trees, samples, vocab, tag_to_ind_map, n_jobs, iterations=200, subset_size=500, verbose=0):
    clf = OneVsRestClassifier(linear_model.SGDClassifier(alpha=0.1, penalty='l2', verbose=verbose, n_jobs=n_jobs))
    X, y = extract_features(trees, samples, vocab, None, tag_to_ind_map)
    clf.fit(X, y)
    return clf


def multilabel_model(trees, samples, vocab, tag_to_ind_map, n_jobs, subset_size=500, verbose=0):
    clf_1 = OneVsRestClassifier(linear_model.SGDClassifier(alpha=0.1, penalty='l2', verbose=verbose, n_jobs=n_jobs))
    clf_2 = OneVsRestClassifier(linear_model.SGDClassifier(alpha=0.1, penalty='l2', verbose=verbose, n_jobs=n_jobs))
    clf_3 = SVC(kernel='rbf', verbose=verbose)
    X, y = extract_features(trees, samples, vocab, None, tag_to_ind_map)
    y_1 = np.array([ind_toaction_map[i].split('-')[0] for i in y])
    y_2 = np.array([ind_toaction_map[i].split('-')[1] if ind_toaction_map[i] != 'SHIFT' else 'SHIFT' for i in y])
    y_3 = np.array([ind_toaction_map[i].split('-')[2] if ind_toaction_map[i] != 'SHIFT' else 'SHIFT' for i in y])
    clf_1.fit(X, y_1)
    clf_2.fit(X, y_2)
    clf_3.fit(X, y_3)
    return clf_1, clf_2, clf_3
