from relations_inventory import ind_toaction_map
from features import extract_features
import torch.nn as nn
import torch


class Network(nn.Module):

    def __init__(self, n_features, hidden_size, num_classes):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.fc1.weight.data.fill_(1.0)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fc2.weight.data.fill_(1.0)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        return nn.functional.relu(self.fc2(x))


def neural_network_model(trees, samples, vocab, tag_to_ind_map, num_iters=200):
    num_classes = len(ind_toaction_map)
    x_train, y_train = extract_features(trees, samples, vocab, None, tag_to_ind_map)
    net = Network(len(x_train[0]), hidden_size=128, num_classes=num_classes)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    for i in range(num_iters):
        y_pred = net(torch.autograd.Variable(torch.tensor(x_train)))
        var = torch.autograd.Variable(torch.tensor(y_train))
        loss = criterion(y_pred, var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net


def neural_net_predict(net, X):
    return net(torch.autograd.Variable(torch.tensor(X, dtype=torch.float)))
