import numpy as np
from preprocess import load_trees
from train_data import gen_train_data
from rst_parser import parse_files
from vocabulary import gen_vocabulary
from pathlib import Path
import argparse
import models


np.random.seed(42)
DATASET_PATH = Path('data')
TRAINING_DIR = DATASET_PATH / 'TRAINING'
DEV_TEST_DIR = DATASET_PATH / 'DEV'
DEV_TEST_GOLD_DIR = DATASET_PATH / 'dev_gold'
PRED_OUTDIR = DATASET_PATH / 'pred'

MODELS = {
    'rnn': models.rnn_model,
    'neural': models.neural_network_model,
    'linear_svm': models.svm_model,
    'random_forest': models.random_forest_model,
    'sgd': models.sgd_model,
    'multi_label': models.multilabel_model,
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=__package__)
    parser.add_argument('--model', choices=MODELS.keys(), default='sgd')
    args = parser.parse_args()

    print('preprocessing..')
    trees = load_trees(TRAINING_DIR)
    vocab, tag_to_ind_map = gen_vocabulary(trees)

    print('training..')
    samples = gen_train_data(trees)
    model = MODELS[args.model](trees, samples, vocab, tag_to_ind_map)

    print('evaluate..')
    dev_trees = load_trees(DEV_TEST_DIR, DEV_TEST_GOLD_DIR)
    parse_files(args.model, model, dev_trees, vocab, tag_to_ind_map,
                DEV_TEST_DIR, DEV_TEST_GOLD_DIR, PRED_OUTDIR)
