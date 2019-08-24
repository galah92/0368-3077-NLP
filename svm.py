from features import extract_features
from sklearn.svm import LinearSVC


def svm(trees, samples, vocab, tag_to_ind_map, verbose=0):
    clf = LinearSVC(penalty='l1', dual=False, tol=1e-7, verbose=verbose)
    x_train, y_train = extract_features(trees, samples, vocab,
                                        None, tag_to_ind_map)
    clf.fit(x_train, y_train)
    return clf
