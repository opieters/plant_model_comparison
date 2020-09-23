import numpy as np
import pandas as pd
import rtoml as toml

def parse_output_data(fn):
    with open(fn) as f:
        header = f.readline()
        header = header.strip().split(",")

        data = dict()

        for line in f:
            sample_data = line.strip().split(",")

            if sample_data[0] == "0":
                for idx, i in enumerate(header[6:]):
                    key = "{}_plant_{}_axis_{}_metamer_{}_organ_{}_element_{}".format(i, *sample_data[1:6])
                    data[key] = [sample_data[ 6 +idx]]
            else:
                for idx, i in enumerate(header[6:]):
                    key = "{}_plant_{}_axis_{}_metamer_{}_organ_{}_element_{}".format(i, *sample_data[1:6])
                    if key in data:
                        data[key].append(sample_data[ 6 +idx])
                    else:
                        data[key] = [0 ] *int(sample_data[0]) + [sample_data[idx]]

    data["t"] = np.arange(int(sample_data[0] ) +1)

    df = pd.DataFrame(data)

    for i in df:
        df[i] = pd.to_numeric(df[i], errors='coerce')

    return df

def parse_input_data(fn, temp_base=4.5, idx_range=(0, 2500)):
    df_in = pd.read_csv(fn)
    if idx_range[0] > 0:
        df_in.drop(index=np.arange(0, idx_range[0], dtype=np.int), axis=0, inplace=True)
    df_in.drop(index=np.arange(idx_range[1], len(df_in), dtype=np.int), axis=0, inplace=True)
    df_in.drop(["t"], axis=1, inplace=True)

    gdd = np.copy(df_in["air_temperature"].values)
    gdd[gdd < temp_base] = temp_base
    gdd = np.array([np.sum(gdd[:i + 1]) for i in range(len(gdd))])
    df_in["GDD"] = gdd / 24

    return df_in


def create_variable_file(df, fn_variables):
    keys = df.keys()
    data = dict()
    for k in keys:
        data[k] = {"category": "general", "description": "", "unit": ""}
    with open(fn_variables, "w") as f:
        toml.dump(data, f)


def load_cplantbox_data(fn):
    df = pd.read_csv(fn, sep="\t")

    return df


