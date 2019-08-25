from relations import ACTIONS
import numpy as np
import torch.nn as nn
import torch


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


def one_hot_encode(label, labels):
    vec = np.zeros(len(labels), dtype=np.float32)
    vec[np.where(labels == ACTIONS[label])] = 1.0
    return vec


def add_padding(X, shape, one_hot=False, labels=None):
    arr = np.zeros(shape=shape)
    x_len = len(X)
    for i in range(shape[0]):
        if i < x_len:
            if one_hot:
                arr[i] = one_hot_encode(X[i], labels)
            else:
                arr[i] = X[i]
    return arr


def rnn(x_train, y_train, **kwargs):
    unique_labels = np.unique([sample.action for sample in kwargs['samples']])
    max_seq_len = max(tree._root.span[1] for tree in kwargs['trees'])
    input_size = len(x_train[0])
    output_size = len(unique_labels)
    model = RnnModel(input_size=input_size, output_size=output_size,
                     hidden_dim=256, n_layers=2, unique_labels=unique_labels,
                     max_seq_len=max_seq_len)
    model.to(device)

    # RNN hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # train data
    sents_idx = kwargs['sents_idx']
    batch_size = sents_idx.count('')
    input_seq = np.zeros((batch_size, max_seq_len, input_size))
    target_seq = np.zeros((batch_size, max_seq_len, output_size))

    old_idx = 0
    j = -1
    for idx in range(len(y_train)):
        if sents_idx[idx] == '':
            j += 1
            input_seq[j] = add_padding(x_train[old_idx:idx], shape=(max_seq_len, input_size))
            target_seq[j] = add_padding(y_train[old_idx:idx], shape=(max_seq_len, output_size), one_hot=True, labels=unique_labels)
            old_idx = idx
    input_seq = torch.from_numpy(input_seq)
    target_seq = torch.Tensor(target_seq)

    # Training
    n_epochs = 100
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        input_seq = input_seq.to(device)
        output, hidden = model(input_seq)
        loss = criterion(output, np.argmax(target_seq, axis=2).view(-1).long())
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch + 1}/{n_epochs}.............', end=' ')
            print(f'Loss: {loss.item():.4f}')
    return model


def rnn_predict(model, input):
    input = torch.from_numpy(input)
    input.to(device)
    out, hidden = model(input)
    prob = nn.functional.softmax(out, dim=0).data
    idx = torch.max(prob, dim=-1)[1]
    return [ACTIONS[i] for i in idx], hidden


class RnnModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers, unique_labels, max_seq_len):
        super(RnnModel, self).__init__()
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
