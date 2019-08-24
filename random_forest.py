from sklearn.ensemble import RandomForestClassifier, BaggingClassifier


def random_forest(x_train, y_train, verbose=0, **kwargs):
    n_estimators = 10
    random_forest = RandomForestClassifier(n_estimators=100, verbose=verbose)
    clf = BaggingClassifier(random_forest,
                            max_samples=1.0 / n_estimators,
                            n_estimators=n_estimators,
                            n_jobs=-1)
    # TODO : Eyal, add SelectFromModel for feature reduction.
    clf.fit(x_train, y_train)
    return clf
