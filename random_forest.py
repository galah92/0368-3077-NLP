from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from features import extract_features


def random_forest(trees, samples, vocab, tag_to_ind_map, verbose=0):
    n_estimators = 10
    random_forest = RandomForestClassifier(n_estimators=100, verbose=verbose)
    clf = BaggingClassifier(random_forest,
                            max_samples=1.0 / n_estimators,
                            n_estimators=n_estimators,
                            n_jobs=-1)
    # TODO : Eyal, add SelectFromModel for feature reduction.
    x_train, y_train = extract_features(trees, samples, vocab,
                                        None, tag_to_ind_map)
    clf.fit(x_train, y_train)
    return clf
