import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import matplotlib


if __name__ == "__main__":
    #matplotlib.use('TkAgg')
    n_plants = 10

    reservoirs = ['respiration+photosynthesis', 'sucrose+fructan+starch', 'proteins', 'N', 'structure', 'cytokinins']
    inputs = ["air_temperature", "PARi", "soil_temperature", "humidity", "Wind", ] # "GDD"

    reservoir_abbr = {
        'respiration+photosynthesis': 'photo',
        'sucrose+fructan+starch': 'energy',
        'proteins': 'proteins',
        'N': 'N',
        'structure': 'structure',
        'cytokinins': 'cytokinins'
    }
    input_abbr = {
        "air_temperature": "Tair",
        "PARi": "PAR",
        "soil_temperature": "Tsoil",
        "humidity": "RH",
        "Wind": "wind",
        "GDD": "GDD"
    }
    fig, axs = plt.subplots(ncols=len(reservoirs),
                            nrows=len(inputs),
                            sharex='all',
                            sharey='all',
                            gridspec_kw={"wspace": 0.0, "hspace": 0.0})

    dfs = dict()

    for j, input_name in enumerate(inputs):
        for i, reservoir in enumerate(reservoirs):
            ax = axs[j][i]

            fn = f"{reservoir}-{input_name}.csv"
            df = pd.read_csv(os.path.join("../experiments/vegetative_stages_{}_plant{}/outputs/".format(n_plants, "s" if n_plants > 1 else ""), fn), index_col=0)

            if "delay" not in dfs:
                dfs["delay"] = df["delay"]
                dfs["alpha"] = np.log10(df["alpha"])

            delays = list(set(df["delay"].values))
            alphas = list(set(df["alpha"].values))
            alphas.sort()
            print(alphas)

            nmse_data = df["NMSE_test"].values
            nmse_data[nmse_data > 1.0] = 1.0
            dfs[f"{reservoir}-{input_name}"] = nmse_data

            nmse_data = np.transpose(nmse_data)

            nmse_data = nmse_data.reshape((len(delays), len(alphas)), order="F")
            im = ax.imshow(nmse_data,
                      extent=(np.min(delays), np.min(alphas), np.max(delays), np.max(alphas)),
                      origin="lower",
                      vmin=0.0,
                      vmax=1.0,
                      aspect="auto",
                      cmap="hot",)
            if j == 0:
                ax.set_title(reservoir_abbr[reservoir])
            if j == len(inputs) - 1:
                ax.set_xlabel("delay")
            if i == 0:
                ax.set_ylabel(f"{input_abbr[input_name]}\nalpha")

            ax.tick_params(axis=u'both', which=u'both', length=0)


    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()

    dfs = pd.DataFrame(dfs)

    dfs.to_csv("vis-data-{}plants.dat".format(n_plants), index=False, sep=" ")

