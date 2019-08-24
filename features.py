import numpy as np
from relations_inventory import action_to_ind_map
from vocabulary import split_edu_to_tags
from vocabulary import split_edu_to_tokens
from vocabulary import DEFAULT_TOKEN
from vocabulary import get_tag_ind

STATE_SIZE = 3

def extract_features(trees, samples, vocab, subset_size, tag_to_ind_map, rnn=False):
    max_edus = max(tree._root.span[1] for tree in trees)
    x_vecs = []
    y_labels = []
    sents_idx = []
    n = len(samples)
    # n = 150 # DEBUG
    sample_idx = np.random.randint(0, subset_size, subset_size) if subset_size is not None else np.array(range(n))
    
    for i in sample_idx:
        _, vec_feats = add_features_per_sample(samples[i], vocab, max_edus, tag_to_ind_map)
        x_vecs.append(vec_feats)
        y_labels.append(action_to_ind_map[samples[i].action])
        sents_idx.append(samples[i].tree._edu_to_sent_ind[samples[i].state[0]])
    if rnn:
        return x_vecs, y_labels, sents_idx
    return x_vecs, y_labels


def add_features_per_sample(sample, vocab, max_edus, tag_to_ind_map):
    features = {}
    feat_names = []
    split_edus = []
    tags_edus = []
    tree = sample.tree
    for i in range(len(sample.state)):
        edu_ind = sample.state[i]
        if edu_ind > 0:
            split_edus.append(split_edu_to_tokens(tree, edu_ind))
            tags_edus.append(split_edu_to_tags(tree, edu_ind))
        else:
            split_edus.append([''])
            tags_edus.append([''])

    for i in range(STATE_SIZE):
        features[f'QueueStackStatus{i}'] = 1 if sample.state[i] == 0 else 0
        features[f'LastTokenIsSeparator{i}'] = 1 if split_edus[i][-1] in ['.', ','] else 0

    feat_names.append(['BEG-WORD-STACK1', 'BEG-WORD-STACK2', 'BEG-WORD-QUEUE1'])
    feat_names.append(['SEC-WORD-STACK1', 'SEC-WORD-STACK2', 'SEC-WORD-QUEUE1'])
    feat_names.append(['THIR-WORD-STACK1', 'THIR-WORD-STACK2', 'THIR-WORD-QUEUE1'])

    feat_names.append(['BEG-TAG-STACK1', 'BEG-TAG-STACK2', 'BEG-TAG-QUEUE1'])
    feat_names.append(['SEC-TAG-STACK1', 'SEC-TAG-STACK2', 'SEC-TAG-QUEUE1'])
    feat_names.append(['THIR-TAG-STACK1', 'THIR-TAG-STACK2', 'THIR-TAG-QUEUE1'])

    for i in range(0,3):
        add_word_features(features, split_edus, feat_names[i], i)

    for i in range(0,3):
        add_tag_features(features, tags_edus, feat_names[i + 3], i, tag_to_ind_map)

    for i in range(STATE_SIZE):
        for n in [0,1,2,-1,-2]:
            features[f'EduWord{n}-State{i}'] = split_edus[i][n] if abs(n) < len(split_edus[i]) else ""
            features[f'EduTag{n}-State{i}'] = tags_edus[i][n] if abs(n) < len(split_edus[i]) else ""

    feat_names = ['END-WORD-STACK1', 'END-WORD-STACK2', 'END-WORD-QUEUE1']
    add_word_features(features, split_edus, feat_names, -1)

    feat_names = ['END-TAG-STACK1', 'END-TAG-STACK2', 'END-TAG-QUEUE1']
    add_tag_features(features, tags_edus, feat_names, -1, tag_to_ind_map)

    add_edu_features(features, tree, sample.state, split_edus, max_edus)

    vecs = gen_vectorized_features(features, vocab, tag_to_ind_map)
    return features, vecs


def add_word_features(features, split_edus, feat_names, word_loc):
    for i in range(len(split_edus)):
        words = split_edus[i]
        feat = feat_names[i]
        features[feat] = DEFAULT_TOKEN
        if words != ['']:
            # last word or one of the first 3 words
            if word_loc < 0 or len(words) > word_loc:
                features[feat] = words[word_loc]


def add_tag_features(features, tags_edus, feat_names, tag_loc, tag_to_ind_map):
    for i in range(len(tags_edus)):
        tags = tags_edus[i]
        feat = feat_names[i]
        features[feat] = ''
        if tags != ['']:
            if tag_loc < 0 or len(tags) > tag_loc:
                features[feat] = tags[tag_loc]


def add_edu_features(features, tree, edus_ind, split_edus, max_edus):
    feat_names = ['LEN-STACK1', 'LEN-STACK2', 'LEN-QUEUE1']

    for i in range(0, 3):
        feat = feat_names[i]
        if edus_ind[i] > 0:
            features[feat] = len(split_edus[i]) / max_edus
        else:
            features[feat] = 0 

    edu_ind_in_tree = []

    for i in range(0, 3):
        if edus_ind[i] > 0:
            edu_ind_in_tree.append(edus_ind[i]) 
        else:
            edu_ind_in_tree.append(0)

    

    features['DIST-FROM-START-STACK1'] = (edu_ind_in_tree[0] - 1.0) / max_edus
    features['DIST-FROM-END-STACK1'] = \
        (tree._root.span[1] - edu_ind_in_tree[0]) / max_edus

    features['DIST-FROM-START-STACK2'] = (edu_ind_in_tree[1] - 1.0) / max_edus
    features['DIST-FROM-END-STACK2'] = \
        (tree._root.span[1] - edu_ind_in_tree[1]) / max_edus

    features['DIST-FROM-START-QUEUE1'] = (edu_ind_in_tree[2] - 1.0) / max_edus

    features['DIST-STACK1-QUEUE1'] = \
        (edu_ind_in_tree[2] - edu_ind_in_tree[0]) / max_edus 

    features['SpanSize'] = tree._root.span[1]-tree._root.span[0]

    features['SameSen-STACK1-QUEUE1'] = 1 if tree._edu_to_sent_ind[edus_ind[0]] == tree._edu_to_sent_ind[edus_ind[2]] else 0
    features['SameSen-STACK1-STACK2'] = 1 if tree._edu_to_sent_ind[edus_ind[0]] == tree._edu_to_sent_ind[edus_ind[1]] else 0


def gen_vectorized_features(features, vocab, tag_to_ind_map):
    vecs = []
    n_tags = len(tag_to_ind_map) - 1
    for key, val in features.items():
        if 'word' in key.lower():
            word_ind = vocab.tokens.get(val.lower(), vocab.tokens[DEFAULT_TOKEN])
            vecs += [elem for elem in vocab.words[word_ind]]
        elif 'tag' in key.lower():
            vecs += [get_tag_ind(tag_to_ind_map, val) / n_tags]
        else:
            vecs += [val]
    return vecs
