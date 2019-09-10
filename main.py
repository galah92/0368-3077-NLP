from evaluation import eval as evaluate
from trees import load_trees
from samples import get_samples
from rst_parser import parse_files
from vocabulary import Vocabulary
from features import get_features
from models import SGD, SVM, RandomForest, MultiLabel, Neural, RNN, VoteModel

from pathlib import Path
import argparse


DATASET_PATH = Path('data')
TRAINING_DIR = DATASET_PATH / 'TRAINING'
DEV_TEST_DIR = DATASET_PATH / 'DEV'
DEV_TEST_GOLD_DIR = DATASET_PATH / 'dev_gold'
PRED_OUTDIR = DATASET_PATH / 'pred'

MODELS = {
    'sgd': SGD,
    'svm': SVM,
    'random_forest': RandomForest,
    'multilabel': MultiLabel,
    'neural': Neural,
    'rnn': RNN,
    'vote': VoteModel
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=__package__)
    parser.add_argument('--model', choices=MODELS.keys(), default='sgd')
    args = parser.parse_args()

    print('preprocessing')
    trees = load_trees(TRAINING_DIR)
    vocab, samples = Vocabulary(trees), get_samples(trees)
    x_train, y_train, sents_idx = get_features(trees, samples, vocab)

    print('training')
    model = MODELS[args.model](trees=trees,
                               samples=samples,
                               sents_idx=sents_idx,
                               n_features=len(x_train[0]),
                               models=[SGD, MultiLabel, RandomForest])
    model.train(x_train, y_train)

    print('evaluating')
    dev_trees = load_trees(DEV_TEST_DIR, DEV_TEST_GOLD_DIR)
    parse_files(model, dev_trees, vocab, DEV_TEST_DIR, PRED_OUTDIR)
    evaluate(DEV_TEST_GOLD_DIR, PRED_OUTDIR)
