from preprocess import preprocess, print_trees_stats, Node, TreeInfo
from train_data import gen_train_data
from rst_parser import parse_files
from vocabulary import gen_vocabulary
from model import mini_batch_linear_model, neural_network_model
from pathlib import Path


# TODO: (Gal)
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


CWD = Path.cwd()
TRAINING_DIR = "TRAINING"
DEV_TEST_DIR = "DEV"
DEV_TEST_GOLD_DIR = "dev_gold"
PRED_OUTDIR = "pred"


if __name__ == '__main__':

    model_name = "neural"  # or "linear"
    baseline = False

    print("preprocessing..")
    trees, max_edus = preprocess(CWD, TRAINING_DIR)
    vocab, tag_to_ind_map = gen_vocabulary(trees)

    if not baseline:
        print("training..")
        samples, y_all = gen_train_data(trees, CWD)
        if model_name == "neural":
            model = neural_network_model(trees,
                                         samples,
                                         vocab,
                                         max_edus,
                                         tag_to_ind_map)
        else:
            model = mini_batch_linear_model(trees,
                                            samples,
                                            y_all,
                                            vocab,
                                            max_edus,
                                            tag_to_ind_map)

    print("evaluate..")
    dev_trees, _ = preprocess(CWD, DEV_TEST_DIR, DEV_TEST_GOLD_DIR)
    parse_files(CWD,
                model_name,
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
