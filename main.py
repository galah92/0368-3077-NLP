from preprocess import preprocess
from train_data import gen_train_data
from rst_parser import parse_files
from vocabulary import gen_vocabulary
from model import mini_batch_linear_model, neural_network_model
from pathlib import Path


DATASET_PATH = Path('data')
TRAINING_DIR = DATASET_PATH / 'TRAINING'
DEV_TEST_DIR = DATASET_PATH / 'DEV'
DEV_TEST_GOLD_DIR = DATASET_PATH / 'dev_gold'
PRED_OUTDIR = DATASET_PATH / 'pred'


if __name__ == '__main__':

    model_name = 'neural'  # or 'linear'
    baseline = False

    print('preprocessing..')
    trees, max_edus = preprocess(TRAINING_DIR)
    vocab, tag_to_ind_map = gen_vocabulary(trees)

    if not baseline:
        print('training..')
        samples, y_all = gen_train_data(trees)
        if model_name == 'neural':
            model = neural_network_model(trees,
                                         samples,
                                         vocab,
                                         max_edus,
                                         tag_to_ind_map,
                                         iterations=10)
        else:
            model = mini_batch_linear_model(trees,
                                            samples,
                                            y_all,
                                            vocab,
                                            max_edus,
                                            tag_to_ind_map)

    print('evaluate..')
    dev_trees, _ = preprocess(DEV_TEST_DIR, DEV_TEST_GOLD_DIR)
    parse_files(model_name,
                model,
                dev_trees,
                vocab,
                max_edus,
                y_all,
                tag_to_ind_map,
                baseline,
                DEV_TEST_DIR,
                DEV_TEST_GOLD_DIR,
                PRED_OUTDIR)
