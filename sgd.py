from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model


def sgd(x_train, y_train, verbose=0, **kwargs):
    sgd = linear_model.SGDClassifier(alpha=0.1, penalty='l2',
                                     n_jobs=-1, verbose=verbose)
    clf = OneVsRestClassifier(sgd)
    clf.fit(x_train, y_train)
    return clf
