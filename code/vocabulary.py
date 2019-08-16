from glove import get_word_vectors
import nltk


DEFAULT_TOKEN = ''


class Vocab():
    def __init__(self):
        self.tokens = {DEFAULT_TOKEN: 0}
        self.words = []


def gen_vocabulary(trees):
    vocab = Vocab()
    vocab.tokens = [DEFAULT_TOKEN]
    vocab.tokens += [word.lower()
                     for tree in trees
                     for edu in tree._EDUS_table
                     for word in nltk.word_tokenize(edu)]
    vocab.tokens = {word: i for i, word in enumerate(set(vocab.tokens))}
    vocab.words = get_word_vectors(vocab.tokens)
    return vocab, build_tags_dict(trees)


def split_edu_to_tokens(tree, edu_ind):
    word_tag_list = tree._edu_word_tag_table[edu_ind]
    return [word for word, _ in word_tag_list]


def split_edu_to_tags(tree, edu_ind):
    word_tag_list = tree._edu_word_tag_table[edu_ind]
    return [tag for _, tag in word_tag_list]


def get_tag_ind(tag_to_ind_map, tag, use_def_tag=False):
    if tag not in tag_to_ind_map:
        if not use_def_tag:
            raise Exception("Could not find tag:" + tag)
        return tag_to_ind_map['']  # empty string treated as default tag
    return tag_to_ind_map[tag]


def build_tags_dict(trees):
    tag_to_ind_map = {'': 0}
    tag_ind = 1
    for tree in trees:
        for word_tag_list in tree._edu_word_tag_table[1:]:
            for _, tag in word_tag_list:
                if tag_to_ind_map.get(tag, None) is None:
                    tag_to_ind_map[tag] = tag_ind
                    tag_ind += 1
    return tag_to_ind_map
