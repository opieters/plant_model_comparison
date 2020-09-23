import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, LeavePGroupsOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def process_data(df_in, df_out, incl_th = 0.95):
    input_names = list(df_in.keys())
    output_names = list(df_out.keys())

    df = pd.concat([df_in, df_out], axis=1, sort=False)
    na_vector = pd.Series([np.nan] * len(df))

    del_keys = []
    for i in df:
        if (df[i].count() / len(df)) < incl_th:
            del_keys.append(i)
    print(f"Deleting {len(del_keys)} keys.")
    df.drop(labels=del_keys, inplace=True, axis=1)

    df.dropna(axis=0, inplace=True)

    for i in del_keys:
        output_names.remove(i)

    return df, output_names, input_names


def nmse_loss(y_true, y_pred):
    return np.mean(np.power(y_pred - y_true, 2.0)) / np.var(y_true)


def get_model():
    sc_params = dict()
    en_params = {"fit_intercept": True,
                 "normalize": False,
                 "precompute": False,
                 "random_state": 5}
    param_grid = {"estimator__alpha": np.power(10., np.arange(-5,6))}
    pca_params = {"n_components": 0.95, "svd_solver": "full"}
    split_params = {"n_groups": 2}

    model = Pipeline([('scaler', StandardScaler(**sc_params)),
                      ('reducer', PCA(**pca_params)),
                      ('estimator', Lasso(**en_params))])

    score = make_scorer(nmse_loss, greater_is_better=False)

    selector = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring=score,
                            cv=LeavePGroupsOut(**split_params),
                            n_jobs=4,
                            refit=True)

    return model, selector


def get_masks(length, subset_size=0.2, train_test_split=(0.7, 0.3)):
    x = np.arange(length, dtype=np.int)
    x = x % int(subset_size*length)
    x[x < int(subset_size*length*train_test_split[0])] = 0
    x[x > 0] = 1
    return x == 0, x == 1
