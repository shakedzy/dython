import pytest
import numpy as np
import pandas as pd
from sklearn import datasets

@pytest.fixture
def iris_df():
    # Use iris dataset as example when needed.
    # Add one made-up categorical column to create a nom-nom relationship.

    iris = datasets.load_iris()

    target = ['C{}'.format(i) for i in iris.target]

    rng = np.random.default_rng(2207)
    extra = rng.choice(list('ABCDE'), size = len(target))

    extra = pd.DataFrame(data=extra, columns=['extra'])

    X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    y = pd.DataFrame(data=target, columns=['target'])

    df = pd.concat([X, extra, y], axis=1)

    return df
