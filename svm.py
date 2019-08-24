from sklearn.svm import LinearSVC


def svm(x_train, y_train, verbose=0, **kwargs):
    clf = LinearSVC(penalty='l1', dual=False, tol=1e-7, verbose=verbose)
    clf.fit(x_train, y_train)
    return clf
