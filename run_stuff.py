import pandas as pd
from dython.nominal import *
from dython.utils import *

df = pd.read_csv('/Users/shakedzy/Desktop/ys.csv')
y_true = df[['true_0','true_1']].as_matrix()
y_pred = df[['pred_0','pred_1']].as_matrix()
y_t = [np.argmax(x) for x in y_true]
y_p = [x[1] for x in y_pred]

roc_graph(y_t,y_p)