#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle, os
from src.analyse.model import process_data, get_model, get_masks, nmse_loss
from src.analyse.load_data import parse_input_data, parse_output_data
import rtoml as toml
from argparse import ArgumentParser
import warnings
warnings.simplefilter('error', RuntimeWarning)


def target(x, d, alpha):
    h = np.arange(d)
    h = np.exp(-np.power((h - (d - 1) / 2), 2))
    h = h / np.sum(h)

    original_length = len(x)

    if d > 0:
        x = np.concatenate((np.zeros((d,)), x))
    x = np.tanh(alpha * x) / np.tanh(alpha)

    y = np.zeros(x.shape)

    if d > 0:
        for i in range(d, y.shape[0]):
            y[i] = x[i - 1] - np.sum(y[i - d:i] * h)

        y = y[d//2:]
        y = y[:original_length]
    else:
        y = x
    return y


def grid_analysis(delays, alphas, X, y, std_y=True):
    analysis_data = dict()

    analysis_data["delay"] = []
    analysis_data["alpha"] = []
    analysis_data["NMSE_mean"] = []
    analysis_data["NMSE_test"] = []
    analysis_data["NMSE_train"] = []
    #analysis_data["params"] = []

    if std_y:
        y = (y - np.mean(y)) / np.std(y)

    for delay in delays:
        train_mask, test_mask = get_masks(len(X), subset_size=0.1)
        X_train, X_test = X[train_mask], X[test_mask]
        for alpha in alphas:
            print(delay, alpha)
            y_tf = target(y, delay, alpha)
            y_train, y_test = y_tf[train_mask], y_tf[test_mask]

            _, selector = get_model()

            groups = np.arange(len(X_train), dtype=np.int) // (len(X_train) // 10)

            selector.fit(X_train, y_train, groups=groups)

            yh = selector.predict(X_test)
            test_error = nmse_loss(y_test, yh)

            yh = selector.predict(X_train)
            train_error = nmse_loss(y_train, yh)

            yh = selector.predict(X)
            mean_error = nmse_loss(y_tf, yh)

            analysis_data["alpha"].append(alpha)
            analysis_data["delay"].append(delay)
            analysis_data["NMSE_train"].append(train_error)
            analysis_data["NMSE_test"].append(test_error)
            analysis_data["NMSE_mean"].append(mean_error)
            #analysis_data["params"].append(selector.best_params_)

    analysis_data = pd.DataFrame(analysis_data)
    return analysis_data


def main(reservoir, input_name, delays, alphas, plant_number):
    variable_info = toml.load(open("variables.toml"))

    analysis_dir = "data/Vegetative_stages/analysis-{}plants/".format(plant_number)
    if not os.path.isdir(analysis_dir):
        os.makedirs(analysis_dir)

    categories = reservoir.split("+")
    variables = []
    for k in variable_info:
        if variable_info[k]["category"] in categories:
            variables.append(k)
    selected_variables = []
    for v in variables:
        for o in output_names:
            if (v + "_") in o:
                selected_variables.append(o)

    if len(selected_variables) < 10:
        print(f"Not enough variables for category {reservoir} (only {len(selected_variables)} variables).")
        return

    X = df.loc[:, selected_variables]
    y = df[input_name].values

    print(reservoir, input_name)
    analysis_data = grid_analysis(delays=delays, alphas=alphas, X=X, y=y)

    print("Saving CSV file.")
    analysis_data.to_csv(os.path.join(analysis_dir, f"{reservoir}-{input_name}.csv"))


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("-r", "--reservoir", type=str, default='cytokinins')
    p.add_argument("-i", "--input", type=str, default='GDD')
    p.add_argument("-n", "--number", type=int, default=1)

    args = p.parse_args()

    reservoirs = ['respiration+photosynthesis', 'sucrose+fructan+starch', 'ignore', 'proteins', 'N', 'structure',
                  'cytokinins']

    fn = "data/cache-{}plants.pkl".format(args.number)
    if os.path.isfile(fn):
        df, output_names, input_names = pickle.load(open(fn, "rb"))
    else:
        df_out = parse_output_data("../experiments/vegetative_stages_{}_plant{}/outputs/elements_outputs.csv".format(args.number, "s" if args.number > 1 else ""))
        df_in = parse_input_data("../experiments/vegetative_stages_{}_plant{}/inputs/meteo_Ljutovac2002.csv".format(args.number, "s" if args.number > 1 else ""),
                                 temp_base=4.5, idx_range=(0, len(df_out)))
        df, output_names, input_names = process_data(df_in, df_out, incl_th = 0.95)

        pickle.dump((df, output_names, input_names), open(fn, "wb"))

    input_names.remove("ambient_CO2")
    input_names.remove("hour")
    input_names.remove("DOY")

    if args.input not in input_names:
        raise ValueError("Input unknown.")
    if args.reservoir not in reservoirs:
        raise ValueError("Reservoir unknown")

    delays = np.arange(0,50,5, dtype=np.int)
    alphas = np.power(10, np.arange(-3.5, 1.5, 0.5))

    if args.reservoir == "ignore":
        exit()
    main(reservoir=args.reservoir, input_name=args.input, delays=delays, alphas=alphas, plant_number=args.number)






