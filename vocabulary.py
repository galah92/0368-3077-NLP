from glove import get_word_vectors
import nltk


DEFAULT_TOKEN = ''


class Vocab():
    def __init__(self):
        self.tokens = {DEFAULT_TOKEN: 0}
        self.words = []


def gen_vocabulary(trees):
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    vocab = Vocab()
    vocab.tokens = [DEFAULT_TOKEN]
    vocab.tokens += [word.lower()
                     for tree in trees
                     for edu in tree._EDUS_table
                     for word in nltk.word_tokenize(edu)]
    vocab.tokens = {word: i for i, word in enumerate(set(vocab.tokens))}
    vocab.words = get_word_vectors(vocab.tokens)
    vocab.tag_to_idx = build_tags_dict(trees)
    return vocab


def split_edu_to_tokens(tree, edu_ind):
    word_tag_list = tree._edu_word_tag_table[edu_ind]
    return [word for word, _ in word_tag_list]


def split_edu_to_tags(tree, edu_ind):
    word_tag_list = tree._edu_word_tag_table[edu_ind]
    return [tag for _, tag in word_tag_list]


def get_tag_ind(tag_to_idx, tag, use_def_tag=False):
    if tag not in tag_to_idx:
        if not use_def_tag:
            raise Exception(f'Could not find tag: {tag}')
        return tag_to_idx['']  # empty string treated as default tag
    return tag_to_idx[tag]


def build_tags_dict(trees):
    tag_to_idx = {'': 0}
    tag_ind = 1
    for tree in trees:
        for word_tag_list in tree._edu_word_tag_table[1:]:
            for _, tag in word_tag_list:
                if tag_to_idx.get(tag, None) is None:
                    tag_to_idx[tag] = tag_ind
                    tag_ind += 1
    return tag_to_idx
