from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#INCOMPLETE
class LeavesToClassifiers:
    _leaves = dict()
    _main_classifier = None
    _single_tree = True

    def __init__(self, classifier, trees=1, **kwargs):
        if trees < 1:
            raise ValueError('Number of trees must be a positive integer')
        elif trees == 1:
            self._single_tree = True
            self._main_classifier = DecisionTreeClassifier(**kwargs)
        else:
            self._single_tree = False
            self._main_classifier = RandomForestClassifier(n_estimators=trees, **kwargs)