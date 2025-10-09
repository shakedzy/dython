from sklearn.ensemble import RandomForestClassifier

from dython.model_utils import random_forest_feature_importance


def test_random_forest_feature_importance_check_types(iris_df):
    X = iris_df.drop(["target", "extra"], axis=1)
    y = iris_df["target"].values

    clf = RandomForestClassifier(n_estimators=7)
    clf.fit(X, y)

    result = random_forest_feature_importance(clf, X.columns)

    assert isinstance(result, list)
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][0], float)
    assert isinstance(result[0][1], str)
