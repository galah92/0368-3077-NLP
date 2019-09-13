RELATIONS = [
    'ATTRIBUTION',
    'BACKGROUND',
    'CAUSE',
    'COMPARISON', 
    'CONDITION',
    'CONTRAST',
    'ELABORATION',
    'ENABLEMENT',
    'TOPICCOMMENT',
    'EVALUATION',
    'EXPLANATION',
    'JOINT',
    'MANNERMEANS',
    'SUMMARY',
    'TEMPORAL',
    'TOPICCHANGE',
    'SPAN',
    'SAME-UNIT',
    'TEXTUALORGANIZATION'
]

NUCLARITIES = ['NN', 'NS', 'SN']

ACTIONS = ['REDUCE-' + nuc + '-' + relation
           for relation in RELATIONS
           for nuc in NUCLARITIES] + ['SHIFT']


def get_features(trees, samples, vocab, max_edus):
    # samples = samples[:150]  # debug
    x_train = [add_features_per_sample(sample, vocab, max_edus)[1]
               for sample in samples]
    y_train = [ACTIONS.index(sample.action) for sample in samples]
    sents_idx = [sample.tree.sents_idx[sample.state[0]] for sample in samples]
    return x_train, y_train, sents_idx


def add_features_per_sample(sample, vocab, max_edus):
    features_dict = {}
    edus = []
    tags = []
    for i in sample.state:
        edus.append([''] if i == 0 else [word for word, _ in sample.tree.pos_tags[i]])
        tags.append([''] if i == 0 else [tag for _, tag in sample.tree.pos_tags[i]])

    for idx, state in enumerate(sample.state):
        features_dict[f'QueueStackStatus{idx}'] = 1 if state == 0 else 0
        features_dict[f'LastTokenIsSeparator{idx}'] = 1 if edus[idx][-1] in ['.', ',',';','"', "'"] else 0
        for n in [0, 1, 2, 3, -1, -2, -3]:
            features_dict[f'EduWord{n}-State{idx}'] = edus[idx][n] if abs(n) < len(edus[idx]) else ""
            features_dict[f'EduTag{n}-State{idx}'] = tags[idx][n] if abs(n) < len(edus[idx]) else ""

    add_edu_features(features_dict, sample.tree, sample.state, edus, max_edus)

    vecs = vectorize_features(features_dict, vocab)
    return features_dict, vecs


def add_word_features(vocab, features_dict, edus, feat_names, word_loc):
    for i in range(len(edus)):
        words = edus[i]
        feat = feat_names[i]
        features_dict[feat] = vocab.DEFAULT_TOKEN
        if words != ['']:
            if word_loc < 0 or len(words) > word_loc:
                features_dict[feat] = words[word_loc]


def add_tag_features(features_dict, tags_list, feat_names, tag_loc):
    for i in range(len(tags_list)):
        tags = tags_list[i]
        feat = feat_names[i]
        features_dict[feat] = ''
        if tags != ['']:
            if tag_loc < 0 or len(tags) > tag_loc:
                features_dict[feat] = tags[tag_loc]


def add_edu_features(features_dict, tree, state, edus, max_edus):
    features = ['LEN-STACK1', 'LEN-STACK2', 'LEN-QUEUE1']

    for idx, i in enumerate(state):
        features_dict[features[idx]] = 0 if i == 0 else len(edus[idx]) / max_edus

    features_dict['DIST-FROM-START-STACK1'] = (state[0] - 1.0) / max_edus
    # features_dict['DIST-FROM-END-STACK1'] = (tree.root.span[1] - state[0]) / max_edus
    features_dict['DIST-FROM-START-STACK2'] = (state[1] - 1.0) / max_edus
    # features_dict['DIST-FROM-END-STACK2'] = (tree.root.span[1] - state[1]) / max_edus
    features_dict['DIST-FROM-START-QUEUE1'] = (state[2] - 1.0) / max_edus
    features_dict['DIST-STACK1-QUEUE1'] = (state[2] - state[0]) / max_edus
    # features_dict['SpanSize'] = tree.root.span[1]-tree.root.span[0]
    features_dict['SameSen-STACK1-QUEUE1'] = 1 if tree.sents_idx[state[0]] == tree.sents_idx[state[2]] else 0
    features_dict['SameSen-STACK1-STACK2'] = 1 if tree.sents_idx[state[0]] == tree.sents_idx[state[1]] else 0


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
