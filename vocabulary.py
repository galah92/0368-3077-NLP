from pathlib import Path
import urllib.request
import zipfile
import numpy as np
import nltk
import sys

from pytorch_pretrained_bert.tokenization import BertTokenizer


class Vocabulary():

    DEFAULT_TOKEN = ''

    def __init__(self, trees):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokens = [Vocabulary.DEFAULT_TOKEN]
        self.tokens += [word.lower()
                        for tree in trees
                        for edu in tree.edus
                        for word in tokenizer.tokenize(edu)]
        self.tokens = {word: i for i, word in enumerate(set(self.tokens))}
        self.words = self._get_word_vectors(self.tokens)
        self.tag_to_idx = self._build_tags_dict(trees)

    def _build_tags_dict(self, trees):
        tag_to_idx = {'': 0}
        tag_ind = 1
        for tree in trees:
            for word_tag_list in tree.pos_tags[1:]:
                for _, tag in word_tag_list:
                    if tag_to_idx.get(tag, None) is None:
                        tag_to_idx[tag] = tag_ind
                        tag_ind += 1
        return tag_to_idx

    def _get_word_vectors(self, tokens):
#         dic = np.load('../my_dict2.npy',allow_pickle=True).item()
#         word_vectors = np.zeros((len(tokens), 768))
        dic = np.load('my_dict.npy',allow_pickle=True).item()
        word_vectors = np.zeros((len(tokens), 50))

        for token in tokens.keys():
            if token == '':
                continue
            word_vectors[tokens[token]] = dic[token]
        return word_vectors