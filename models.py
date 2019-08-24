from sklearn import linear_model
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from relations_inventory import ind_toaction_map, action_to_ind_map
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


def one_hot_encode(label, labels, action_to_ind_map=action_to_ind_map):
    vec = np.zeros(len(labels), dtype=np.float32)
    vec[np.where(labels == ind_toaction_map[label])] = 1.0
    return vec

def add_padding(X, shape, one_hot=False, labels=None):
    arr = np.zeros(shape=shape)
    max_len = shape[0]
    x_len = len(X)

    for i in range(max_len):
        if i < x_len:
            if one_hot:
                arr[i] = one_hot_encode(X[i], labels)
            else:
                arr[i] = X[i]

    return arr


def rnn_model(trees, samples, vocab, tag_to_ind_map):
    unique_labels = np.unique([sample.action for sample in samples])
    max_seq_len = max(tree._root.span[1] for tree in trees)
    output_size = len(unique_labels)
    input_size = len(extract_features(trees, samples, vocab, 1, tag_to_ind_map)[0][0])
    model = RNN_Model(input_size=input_size, output_size=output_size, hidden_dim=256, n_layers=2, unique_labels=unique_labels, max_seq_len=max_seq_len)
    model.to(device)

    # RNN hyperparameters
    n_epochs = 100
    lr=0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train data
    x_vecs, y_labels, sents_idx = extract_features(trees, samples, vocab, None, tag_to_ind_map, rnn=True)
    batch_size = sents_idx.count("")
    input_seq = np.zeros((batch_size, max_seq_len, input_size), dtype=np.float32)
    target_seq = np.zeros((batch_size, max_seq_len, output_size), dtype=np.float32)

    old_idx = 0
    j = -1
    for idx in range(len(y_labels)):
        if sents_idx[idx] == "":
            j += 1
            input_seq[j] = add_padding(x_vecs[old_idx:idx], shape=(max_seq_len, input_size))
            target_seq[j] = add_padding(y_labels[old_idx:idx], shape=(max_seq_len, output_size), one_hot=True, labels=unique_labels)
            old_idx = idx
    input_seq = torch.from_numpy(input_seq)
    target_seq = torch.Tensor(target_seq)

    # Training
    # n_epochs = 2 # DEBUG
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        input_seq = input_seq.to(device)
        output, hidden = model(input_seq)
        loss = criterion(output, np.argmax(target_seq,axis=2).view(-1).long())
        loss.backward() 
        optimizer.step() 
        
        if epoch%10 == 0:
            print (f'Epoch: {epoch}/{n_epochs}.............', end=' ')
            print (f'Loss: {loss.item():.4f}')

    return model


def rnn_predict(model, input):
    input = torch.from_numpy(input)
    input.to(device)
    out, hidden = model(input)
    prob = nn.functional.softmax(out, dim=0).data
    idx = torch.max(prob, dim=-1)[1]

    return [ind_toaction_map[i] for i in idx], hidden


class RNN_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, unique_labels, max_seq_len):
        super(RNN_Model, self).__init__()

        # Defining some parameters
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
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden


def svm_model(trees, samples, vocab, tag_to_ind_map, verbose=0):
    clf = LinearSVC(penalty='l1', dual=False, tol=1e-7, verbose=verbose)

    X, y = extract_features(trees, samples, vocab, None, tag_to_ind_map)
    clf.fit(X, y)
    return clf


def random_forest_model(trees, samples, vocab, tag_to_ind_map, verbose=0):
    n_estimators = 10
    clf = BaggingClassifier(RandomForestClassifier(n_estimators = 100, verbose=verbose),
                            max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1)
    # TODO : Eyal, add SelectFromModel for feature reduction.
    X, y = extract_features(trees, samples, vocab, None, tag_to_ind_map)
    clf.fit(X, y)
    return clf


def neural_network_model(trees, samples, vocab, tag_to_ind_map, iterations=200):
    
    num_classes = len(ind_toaction_map)

    [x_vecs, _] = extract_features(trees, samples, vocab, 1, tag_to_ind_map)

    print("num features {}, num classes {}".format(len(x_vecs[0]), num_classes))
    print("Running neural model")

    net = Network(len(x_vecs[0]), hidden_size, num_classes)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for i in range(iterations):
        [x_vecs, y_labels] = extract_features(trees, samples, vocab, 5000, tag_to_ind_map)
        y_pred = net(Variable(torch.tensor(x_vecs, dtype=torch.float)))
        loss = criterion(y_pred, Variable(torch.tensor(y_labels, dtype=torch.long)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return net


def neural_net_predict(net, x_vecs):
    return net(Variable(torch.tensor(x_vecs, dtype=torch.float)))


def sgd_model(trees, samples, vocab, tag_to_ind_map, iterations=200, verbose=0):
    clf = OneVsRestClassifier(linear_model.SGDClassifier(alpha=0.1, penalty='l2', verbose=verbose, n_jobs=-1))
    X, y = extract_features(trees, samples, vocab, None, tag_to_ind_map)
    clf.fit(X, y)
    return clf

def multilabel_model(trees, samples, vocab, tag_to_ind_map, verbose=0):
    clf_1 = BaggingClassifier(verbose=verbose, n_jobs=-1)
    clf_2 = BaggingClassifier(verbose=verbose, n_jobs=-1)
    clf_3 = SVC(kernel='rbf', verbose=verbose)
    X, y = extract_features(trees, samples, vocab, None, tag_to_ind_map)
    y_1 = np.array([ind_toaction_map[i].split('-')[0] for i in y])
    y_2 = np.array([ind_toaction_map[i].split('-')[1] if ind_toaction_map[i] != 'SHIFT' else 'SHIFT' for i in y])
    y_3 = np.array([ind_toaction_map[i].split('-')[2] if ind_toaction_map[i] != 'SHIFT' else 'SHIFT' for i in y])
    clf_1.fit(X, y_1)
    clf_2.fit(X, y_2)
    clf_3.fit(X, y_3)
    return clf_1, clf_2, clf_3
