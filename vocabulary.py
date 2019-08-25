from pathlib import Path
import urllib.request
import zipfile
import numpy as np
import nltk


class Vocabulary():

    DEFAULT_TOKEN = ''

    def __init__(self, trees):
        self.tokens = [Vocabulary.DEFAULT_TOKEN]
        self.tokens += [word.lower()
                        for tree in trees
                        for edu in tree._EDUS_table
                        for word in nltk.word_tokenize(edu)]
        self.tokens = {word: i for i, word in enumerate(set(self.tokens))}
        self.words = self._get_word_vectors(self.tokens)
        self.tag_to_idx = self._build_tags_dict(trees)

    def _build_tags_dict(self, trees):
        tag_to_idx = {'': 0}
        tag_ind = 1
        for tree in trees:
            for word_tag_list in tree._edu_word_tag_table[1:]:
                for _, tag in word_tag_list:
                    if tag_to_idx.get(tag, None) is None:
                        tag_to_idx[tag] = tag_ind
                        tag_ind += 1
        return tag_to_idx

    def _get_word_vectors(self, tokens):
        glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        glove_dir = Path(__file__).resolve().parent / 'glove'
        glove_zip = glove_dir / 'glove.6B.zip'
        glove_6b_50d = glove_dir / 'glove.6B.50d.txt'
        if not glove_zip.is_file():
            print(f'downloading glove files to {glove_zip}')
            glove_dir.mkdir(exist_ok=True)
            urllib.request.urlretrieve(glove_url, str(glove_zip))
        if not glove_6b_50d.is_file():
            with zipfile.ZipFile(glove_zip, 'r') as f:
                print(f'extracting {glove_zip}')
                f.extractall(str(glove_dir))
        word_vectors = np.zeros((len(tokens), 50))
        with glove_6b_50d.open(encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token, *data = line.split()
                if token not in tokens:
                    continue
                word_vectors[tokens[token]] = [float(x) for x in data]
        return word_vectors
