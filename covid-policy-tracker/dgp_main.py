
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erfinv

import dgp.variants as dgp
from datamodule import ResponseDataModule


def plot_country(model, df, country="Germany", confidence=0.95):
    df = df[df["country"] == country]

    df.pop("country")

    y = df.pop("reproduction_rate").to_numpy()
    x = df.to_numpy()

    # drop index
    x = x[:, 1:]

    y = y[..., None]

    mean, var = model.predict(x)

    std = np.sqrt(var)

    shade_std = np.sqrt(2) * erfinv(confidence)

    lshade = mean - shade_std * std
    ushade = mean + shade_std * std

    lshade = np.squeeze(lshade)
    ushade = np.squeeze(ushade)

    ax = plt.gca()
    ax.plot(y, label="Actual", color="C0")
    ax.plot(mean, label="Predicted", color="C1")
    x = np.linspace(0, len(lshade), num=len(lshade))
    ax.fill_between(x, lshade, ushade, color="C1", alpha=0.2, label=f"{int(100 * confidence):d}% Confidence")
    ax.set_xlabel("Time")
    ax.set_ylabel("R")
    ax.set_title(country)
    ax.legend()


def plot_countries(model, countries=("Germany",)):
    df = pd.read_csv("policies_onehot_full.csv")

    nrows = int(round(np.sqrt(len(countries))))
    ncols = len(countries) // nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols + 1, 6 * nrows))

    for i, country in enumerate(countries):
        plt.subplot(nrows, ncols, i + 1)
        plot_country(model, df, country)

    # set all ylims equal
    ylims = []
    for ax in axes.flat:
        ylims.extend(ax.get_ylim())

    ylims = [min(ylims), max(ylims)]
    for ax in axes.flat:
        ax.set_ylim(ylims)

    plt.savefig("test.png")
    plt.show()


def main():

    dm = ResponseDataModule()
    dm.prepare_data()
    dm.setup()

    df = dm.df

    # df = df[df["country"] == "Germany"]

    df.pop("country")

    y = df.pop("reproduction_rate").to_numpy()
    x = df.to_numpy()

    # drop index
    train_x = x[:, 1:]
    train_y = y[..., None]

    print(train_x.shape)
    print(train_y.shape)

    model = dgp.ProductOfExperts(train_x, train_y, m=20)

    model.optimize()

    countries = ("Germany", "Spain", "Italy", "Japan", "Australia", "Argentina")

    plot_countries(model=model, countries=countries)


def test():

    x = np.linspace(0, 2 * np.pi, 1000)
    y = np.sin(x)

    x = x[:, None]
    y = y[:, None]

    # drop random parts
    train_idx = np.random.choice(len(x), size=20)

    models = [
        dgp.ProductOfExperts,
        dgp.CaoFleetPoE,
        dgp.GeneralisedPoE,
        dgp.BayesianCommitteeMachine,
    ]

    nrows = 2
    ncols = 2

    plt.figure(figsize=(6 * ncols + 1, 6 * nrows))

    for i, model in enumerate(models):
        model = model(x[train_idx], y[train_idx], m=5)
        model.optimize()

        mean, var = model.predict(x)

        std = np.sqrt(var)

        l, u = mean - 1.96 * std, mean + 1.96 * std

        plt.subplot(nrows, ncols, i + 1)
        plt.plot(x, y, color="C0", label="Actual")
        plt.plot(x, mean, color="C1", label="Predicted")
        plt.fill_between(x.flat, l.flat, u.flat, color="C1", label="95% Confidence", alpha=0.2)
        plt.plot(x[train_idx], y[train_idx], linewidth=0, marker="x", color="C2", label="Train Points")
        plt.title(model.__class__.__name__)
        plt.legend()

    plt.show()



if __name__ == "__main__":
    # main()
    test()
