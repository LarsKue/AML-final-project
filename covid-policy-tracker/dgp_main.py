import GPy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erfinv
import pickle

import uncertainties
import uncertainties.unumpy as unp

import dgp.variants as dgp
from datamodule import ResponseDataModule


def inormalize(mean, var):
    m = 1.01562305
    v = 0.34747347

    mean = mean * np.sqrt(v) + m
    var = var * v

    return mean, var


def normalize(y):
    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)

    return (y - mean) / std


global_mean = 0.0
global_std = 1.0


def plot_country(model, df, country="Germany", confidence=0.95):
    print("Plotting Country", country)
    df = df[df["country"] == country]

    df.pop("country")

    y = df.pop("reproduction_rate").to_numpy()
    x = df.to_numpy()

    # drop index
    x = x[:, 1:]

    y = y[..., None]

    mean, var = model.predict(x)

    mean = mean * global_std + global_mean
    var = var * global_std ** 2

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
    df = pd.read_csv("policies_onehot_full_absolute_R.csv")

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

    modelname = model.__class__.__name__
    plt.suptitle(modelname)
    plt.savefig(f"{modelname}.png")
    plt.show()


def main():
    global global_mean
    global global_std

    dm = ResponseDataModule()
    dm.prepare_data()
    dm.setup()

    df = dm.df

    # countries = ("Germany", "Spain", "Italy", "Japan", "Australia", "Argentina")
    #
    # df = df[df["country"].isin(countries)]

    df.pop("country")

    y = df.pop("reproduction_rate").to_numpy()
    x = df.to_numpy()

    # drop index
    train_x = x[:, 1:]
    train_y = y[..., None]

    print(train_x.shape)
    print(train_y.shape)

    # normalize response
    global_mean = np.mean(train_y, axis=0)
    global_std = np.std(train_y, axis=0)

    print("Global Mean:", global_mean)
    print(global_std)

    train_y = (train_y - global_mean) / global_std

    models = [
        dgp.GeneralisedRobustBCM,
        dgp.ProductOfExperts,
        dgp.CaoFleetPoE,
        dgp.GeneralisedPoE,
        dgp.BayesianCommitteeMachine,
        dgp.RobustBCM,
    ]

    def kernel(shape):
        return GPy.kern.RBF(shape) + GPy.kern.White(shape)
        # return GPy.kern.Matern52(shape)

    countries = ("Germany", "Spain", "Italy", "Japan", "Australia", "Argentina")

    for i, model in enumerate(models):
        model = model(train_x, train_y, kernel=kernel, m=32)
        modelname = model.__class__.__name__
        print("Model:", modelname)

        print("Optimizing...")
        model.optimize()

        print("Plotting...")
        plot_countries(model=model, countries=countries)

        with open(f"{modelname}.dump", "wb+") as f:
            pickle.dump(model, f)


def test():

    x = np.linspace(0, 2 * np.pi, 1000)
    y = np.sin(x)

    noise = np.random.normal(0, 0.1, size=y.shape)

    y = y + noise

    x = x[:, None]
    y = y[:, None]

    kernel = GPy.kern.RBF(input_dim=x.shape[1]) + GPy.kern.White(input_dim=x.shape[1])
    full_gaussian = GPy.models.GPRegression(x, y, kernel=kernel)

    full_gaussian.optimize()

    fgx = x
    fgmean, fgvar = full_gaussian.predict(x)
    fgstd = np.sqrt(fgvar)
    fgl, fgu = fgmean - 1.96 * fgstd, fgmean + 1.96 * fgstd

    models = [
        dgp.GeneralisedRobustBCM,
        dgp.ProductOfExperts,
        dgp.CaoFleetPoE,
        dgp.GeneralisedPoE,
        dgp.BayesianCommitteeMachine,
        dgp.RobustBCM,
    ]

    nrows = 3
    ncols = 2

    # plt.figure(figsize=(6 * ncols + 1, 6 * nrows))

    for i, model in enumerate(models):
        model = model(x, y, m=16)
        model.optimize()

        mean, var = model.predict(x)

        std = np.sqrt(var)

        l, u = mean - 1.96 * std, mean + 1.96 * std

        # plt.subplot(nrows, ncols, i + 1)

        plt.figure(figsize=(12, 9))
        plt.plot(fgx, fgmean, color="black", label="Full Gaussian")
        plt.fill_between(fgx.flat, fgl.flat, fgu.flat, color="black", alpha=0.2)
        plt.plot(x, mean, color="C1", label="Predicted")
        plt.fill_between(x.flat, l.flat, u.flat, color="C1", label="95% Confidence", alpha=0.2)
        plt.plot(x, y, linewidth=0, marker="x", color="C2", label="Train Points")
        plt.title(f"{model.__class__.__name__}, {len(model.partitions())} Experts")
        plt.legend()

        plt.xlabel("x")
        plt.ylabel("y")

        plt.savefig(f"method_{model.__class__.__name__}.png")
        plt.close()


def load():
    filename = "GeneralisedRobustBCM.dump"
    with open(filename, "rb") as f:
        model = pickle.load(f)

    p0 = np.zeros((1, 46))
    p = np.concatenate((p0, np.eye(46)), axis=0)

    # effect of policies at 0 vaccination rate
    x = np.array([[0, 0]])
    x = np.repeat(x, len(p), axis=0)

    x = np.concatenate((p, x), axis=-1)

    # mean, var = model.predict(x)
    #
    # np.save("meanR.npy", mean)
    # np.save("varR.npy", var)

    mean, var = np.load("meanR.npy"), np.load("varR.npy")

    mean, var = inormalize(mean, var)

    r = unp.uarray(mean, np.sqrt(var))

    baseline = mean[0]

    reduction = 100.0 * (r - baseline) / baseline

    yticks = np.array([
        "None",

        "C1 School Closings 1",
        "C1 School Closings 2",
        "C1 School Closings 3",

        "C2 Workplace Closings 1",
        "C2 Workplace Closings 2",
        "C2 Workplace Closings 3",

        "C3 Public Event Cancellations 1",
        "C3 Public Event Cancellations 2",

        "C4 Gathering Restrictions 1",
        "C4 Gathering Restrictions 2",
        "C4 Gathering Restrictions 3",
        "C4 Gathering Restrictions 4",

        "C5 Public Transport Closings 1",
        "C5 Public Transport Closings 2",

        "C6 Stay at Home 1",
        "C6 Stay at Home 2",
        "C6 Stay at Home 3",

        "C7 Internal Movement Restrictions 1",
        "C7 Internal Movement Restrictions 2",

        "C8 International Travel Controls 1",
        "C8 International Travel Controls 2",
        "C8 International Travel Controls 3",
        "C8 International Travel Controls 4",

        "E1 Income Support 1",
        "E1 Income Support 2",

        "E2 Debt Relief 1",
        "E2 Debt Relief 2",

        "H1 Public Info Campaigns 1",
        "H1 Public Info Campaigns 2",

        "H2 Testing Policy 1",
        "H2 Testing Policy 2",
        "H2 Testing Policy 3",

        "H3 Contact Tracing 1",
        "H3 Contact Tracing 2",

        "H6 Facial Coverings 1",
        "H6 Facial Coverings 2",
        "H6 Facial Coverings 3",
        "H6 Facial Coverings 4",

        "H7 Vaccination Policy 1",
        "H7 Vaccination Policy 2",
        "H7 Vaccination Policy 3",
        "H7 Vaccination Policy 4",
        "H7 Vaccination Policy 5",

        "H8 Protection of Elderly 1",
        "H8 Protection of Elderly 2",
        "H8 Protection of Elderly 3",
    ])

    reduction = np.squeeze(reduction)

    y = np.array(list(reversed(np.arange(len(reduction)))))

    sort = np.argsort(reduction)

    reduction = reduction[sort]
    yticks = yticks[sort]

    colors = np.where(unp.nominal_values(reduction) < 0, "green", "red")

    colors[np.isclose(unp.nominal_values(reduction), 0)] = "black"

    plt.figure(figsize=(12, 10))
    plt.errorbar(unp.nominal_values(reduction), y, xerr=unp.std_devs(reduction), fmt=".", ecolor=colors)

    plt.axvline(color="black", linewidth=0.5)
    plt.yticks(y, yticks)
    plt.xlabel("Percentage Change in $R_t$")
    plt.tight_layout()

    plt.savefig(f"reduction.png")
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)

    # main()
    # test()
    load()
