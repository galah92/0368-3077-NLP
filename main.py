import numpy as np
from preprocess import load_trees
from train_data import gen_train_data
from rst_parser import parse_files
from vocabulary import gen_vocabulary
from pathlib import Path
import models

np.random.seed(42)
DATASET_PATH = Path('data')
TRAINING_DIR = DATASET_PATH / 'TRAINING'
DEV_TEST_DIR = DATASET_PATH / 'DEV'
DEV_TEST_GOLD_DIR = DATASET_PATH / 'dev_gold'
PRED_OUTDIR = DATASET_PATH / 'pred'


if __name__ == '__main__':

    model_name = 'linear_svm'  # ['neural', 'linear', 'linear_svm', 'random_forest']
    baseline = False

    print('preprocessing..')
    trees = load_trees(TRAINING_DIR)
    vocab, tag_to_ind_map = gen_vocabulary(trees)

    if not baseline:
        print('training..')
        samples, y_all = gen_train_data(trees)
        if model_name == 'neural':
            model = models.neural_network_model(trees,
                                                samples,
                                                vocab,
                                                tag_to_ind_map,
                                                iterations=10)
        
        elif model_name == 'linear_svm':
            model = models.svm_model(trees, samples, y_all, vocab, tag_to_ind_map, n_jobs=1)
        
        elif model_name == 'random_forest':
            model = models.random_forest_model(trees, samples, y_all, vocab, tag_to_ind_map, n_jobs=1)

        elif model_name == 'linear':
            model = models.mini_batch_linear_model(trees,
                                                   samples,
                                                   y_all,
                                                   vocab,
                                                   tag_to_ind_map)

    print('evaluate..')
    dev_trees = load_trees(DEV_TEST_DIR, DEV_TEST_GOLD_DIR)
    parse_files(model_name,
                model,
                dev_trees,
                vocab,
                y_all,
                tag_to_ind_map,
                baseline,
                DEV_TEST_DIR,
                DEV_TEST_GOLD_DIR,
                PRED_OUTDIR)
