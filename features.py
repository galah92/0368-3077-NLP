from relations import ACTIONS


def get_features(trees, samples, vocab):
    # samples = samples[:150]  # debug
    max_edus = max(tree._root.span[1] for tree in trees)
    x_train = [get_features_for_sample(sample, vocab, max_edus)[1]
               for sample in samples]
    y_train = [ACTIONS.index(sample.action) for sample in samples]
    sents_idx = [sample.tree._edu_to_sent_ind[sample.state[0]] for sample in samples]
    return x_train, y_train, sents_idx


def get_features_for_sample(sample, vocab, max_edus):
    features = {}
    feat_names, split_edus, tags_edus = [], [], []
    tree = sample.tree
    for i in range(len(sample.state)):
        edu_ind = sample.state[i]
        if edu_ind > 0:
            split_edus.append(split_edu_to_tokens(tree, edu_ind))
            tags_edus.append(split_edu_to_tags(tree, edu_ind))
        else:
            split_edus.append([''])
            tags_edus.append([''])

    for i in range(3):
        features[f'QueueStackStatus{i}'] = 1 if sample.state[i] == 0 else 0
        features[f'LastTokenIsSeparator{i}'] = 1 if split_edus[i][-1] in ['.', ','] else 0

    feat_names.append(['BEG-WORD-STACK1', 'BEG-WORD-STACK2', 'BEG-WORD-QUEUE1'])
    feat_names.append(['SEC-WORD-STACK1', 'SEC-WORD-STACK2', 'SEC-WORD-QUEUE1'])
    feat_names.append(['THIR-WORD-STACK1', 'THIR-WORD-STACK2', 'THIR-WORD-QUEUE1'])
    feat_names.append(['BEG-TAG-STACK1', 'BEG-TAG-STACK2', 'BEG-TAG-QUEUE1'])
    feat_names.append(['SEC-TAG-STACK1', 'SEC-TAG-STACK2', 'SEC-TAG-QUEUE1'])
    feat_names.append(['THIR-TAG-STACK1', 'THIR-TAG-STACK2', 'THIR-TAG-QUEUE1'])

    for i in range(3):
        add_word_features(vocab, features, split_edus, feat_names[i], i)
        add_tag_features(features, tags_edus, feat_names[i + 3], i)
        for n in [0, 1, 2, -1, -2]:
            features[f'EduWord{n}-State{i}'] = split_edus[i][n] if abs(n) < len(split_edus[i]) else ""
            features[f'EduTag{n}-State{i}'] = tags_edus[i][n] if abs(n) < len(split_edus[i]) else ""

    feat_names = ['END-WORD-STACK1', 'END-WORD-STACK2', 'END-WORD-QUEUE1']
    add_word_features(vocab, features, split_edus, feat_names, -1)
    feat_names = ['END-TAG-STACK1', 'END-TAG-STACK2', 'END-TAG-QUEUE1']
    add_tag_features(features, tags_edus, feat_names, -1)
    add_edu_features(features, tree, sample.state, split_edus, max_edus)

    vecs = vectorize_features(features, vocab)
    return features, vecs


def add_word_features(vocab, features, split_edus, feat_names, word_loc):
    for i in range(len(split_edus)):
        words = split_edus[i]
        feat = feat_names[i]
        features[feat] = vocab.DEFAULT_TOKEN
        if words != ['']:
            # last word or one of the first 3 words
            if word_loc < 0 or len(words) > word_loc:
                features[feat] = words[word_loc]


def add_tag_features(features, tags_edus, feat_names, tag_loc):
    for i in range(len(tags_edus)):
        tags = tags_edus[i]
        feat = feat_names[i]
        features[feat] = ''
        if tags != ['']:
            if tag_loc < 0 or len(tags) > tag_loc:
                features[feat] = tags[tag_loc]


def add_edu_features(features, tree, edus_ind, split_edus, max_edus):
    feat_names = ['LEN-STACK1', 'LEN-STACK2', 'LEN-QUEUE1']
    for i in range(3):
        feat = feat_names[i]
        if edus_ind[i] > 0:
            features[feat] = len(split_edus[i]) / max_edus
        else:
            features[feat] = 0 
    edu_ind_in_tree = []
    for i in range(3):
        if edus_ind[i] > 0:
            edu_ind_in_tree.append(edus_ind[i]) 
        else:
            edu_ind_in_tree.append(0)
    features['DIST-FROM-START-STACK1'] = (edu_ind_in_tree[0] - 1.0) / max_edus
    features['DIST-FROM-END-STACK1'] = (tree._root.span[1] - edu_ind_in_tree[0]) / max_edus
    features['DIST-FROM-START-STACK2'] = (edu_ind_in_tree[1] - 1.0) / max_edus
    features['DIST-FROM-END-STACK2'] = (tree._root.span[1] - edu_ind_in_tree[1]) / max_edus
    features['DIST-FROM-START-QUEUE1'] = (edu_ind_in_tree[2] - 1.0) / max_edus
    features['DIST-STACK1-QUEUE1'] = (edu_ind_in_tree[2] - edu_ind_in_tree[0]) / max_edus
    features['SpanSize'] = tree._root.span[1]-tree._root.span[0]
    features['SameSen-STACK1-QUEUE1'] = 1 if tree._edu_to_sent_ind[edus_ind[0]] == tree._edu_to_sent_ind[edus_ind[2]] else 0
    features['SameSen-STACK1-STACK2'] = 1 if tree._edu_to_sent_ind[edus_ind[0]] == tree._edu_to_sent_ind[edus_ind[1]] else 0


def split_edu_to_tokens(tree, edu_ind):
    word_tag_list = tree._edu_word_tag_table[edu_ind]
    return [word for word, _ in word_tag_list]


def split_edu_to_tags(tree, edu_ind):
    word_tag_list = tree._edu_word_tag_table[edu_ind]
    return [tag for _, tag in word_tag_list]


def vectorize_features(features, vocab):
    vecs = []
    n_tags = len(vocab.tag_to_idx) - 1
    for key, val in features.items():
        if 'word' in key.lower():
            word_ind = vocab.tokens.get(val.lower(), vocab.tokens[vocab.DEFAULT_TOKEN])
            vecs += [elem for elem in vocab.words[word_ind]]
        elif 'tag' in key.lower():
            vecs += [vocab.tag_to_idx[val] / n_tags]
        else:
            vecs += [val]
    return vecs
