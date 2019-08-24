from relations_inventory import ind_toaction_map
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import numpy as np


def multi_label(x_train, y_train, verbose=0, **kwargs):
    clf_1 = BaggingClassifier(verbose=verbose, n_jobs=-1)
    clf_2 = BaggingClassifier(verbose=verbose, n_jobs=-1)
    clf_3 = SVC(kernel='rbf', verbose=verbose)
    y_1 = np.array([ind_toaction_map[i].split('-')[0] for i in y_train])
    y_2 = np.array([ind_toaction_map[i].split('-')[1] if ind_toaction_map[i] != 'SHIFT' else 'SHIFT' for i in y_train])
    y_3 = np.array([ind_toaction_map[i].split('-')[2] if ind_toaction_map[i] != 'SHIFT' else 'SHIFT' for i in y_train])
    clf_1.fit(x_train, y_1)
    clf_2.fit(x_train, y_2)
    clf_3.fit(x_train, y_3)
    return clf_1, clf_2, clf_3
