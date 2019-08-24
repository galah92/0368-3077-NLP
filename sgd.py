from features import extract_features
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model


def sgd(trees, samples, vocab, tag_to_ind_map, verbose=0):
    sgd = linear_model.SGDClassifier(alpha=0.1, penalty='l2',
                                     n_jobs=-1, verbose=verbose)
    clf = OneVsRestClassifier(sgd)
    x_train, y_train = extract_features(trees, samples, vocab,
                                        None, tag_to_ind_map)
    clf.fit(x_train, y_train)
    return clf
