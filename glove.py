from pathlib import Path
import tempfile
import urllib.request
import numpy as np
import zipfile


GLOVE_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'
GLOVE_PATH = Path(tempfile.gettempdir())
GLOVE_ZIP = GLOVE_PATH / 'glove.6B.zip'
GLOVE_6B_50D = GLOVE_PATH / 'glove.6B.50d.txt'


def get_word_vectors(tokens):

    if not GLOVE_ZIP.is_file():
        print(f'downloading glove files to {GLOVE_ZIP}')
        urllib.request.urlretrieve(GLOVE_URL, str(GLOVE_ZIP))

    if not GLOVE_6B_50D.is_file():
        with zipfile.ZipFile(GLOVE_ZIP, 'r') as f:
            print(f'extracting {GLOVE_ZIP}')
            f.extractall(str(GLOVE_PATH))

    word_vectors = np.zeros((len(tokens), 50))
    with GLOVE_6B_50D.open(encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token, *data = line.split()
            if token not in tokens:
                continue
            word_vectors[tokens[token]] = [float(x) for x in data]
    return word_vectors
