
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from datetime import datetime, timedelta

import torch
import torch.nn as nn

import data
from policy_tracker import PolicyTracker


def main():
    responses, responses2, testing = data.fetch()

    # responses2.reencode("cp1252", "utf8")
    #
    # responses.save()
    # responses2.save()
    # testing.save()

    responses.load()
    responses2.load()
    testing.load()

    responses = responses.to_df(low_memory=False)
    responses2 = responses2.to_df()
    testing = testing.to_df()

    print(responses.dtypes)
    print("========================================")
    print(responses2.dtypes)
    print(responses2["Measure_L1"].nunique())
    print(responses2["Measure_L2"].nunique())
    print(responses2["Measure_L3"].nunique())
    print(responses2["Measure_L4"].nunique())

    print(responses2["Measure_L2"])
    print("========================================")
    print(testing.dtypes)

    print(responses.shape)

    # convert measures to one-hot
    lb = LabelBinarizer()

    responses2["L2"] = lb.fit_transform(responses2["Measure_L2"]).tolist()

    print(responses2["L2"])



    # usa_r = responses[responses["CountryName"] == "United States"]
    # usa_t = testing[testing["location"] == "United States"]
    #
    # germany_r = responses[responses["CountryName"] == "Germany"]
    # germany_t = testing[testing["location"] == "Germany"]
    #
    # nrows = 1
    # ncols = 2
    # fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 10 * nrows))
    #
    # axes[0].plot(usa_t["total_vaccinations_per_hundred"], usa_t["new_cases_smoothed_per_million"], label="USA")
    # axes[0].plot(germany_t["total_vaccinations_per_hundred"], germany_t["new_cases_smoothed_per_million"], label="GER")
    # axes[0].set_xlabel("% Fully Vaccinated")
    # axes[0].set_ylabel("New Cases per Million Population")
    # axes[0].legend()
    #
    # axes[1].plot(usa_t["total_vaccinations_per_hundred"], usa_t["new_deaths_smoothed_per_million"], label="USA")
    # axes[1].plot(germany_t["total_vaccinations_per_hundred"], germany_t["new_deaths_smoothed_per_million"], label="GER")
    # axes[1].set_xlabel("% Fully Vaccinated")
    # axes[1].set_ylabel("New Deaths per Million Population")
    # axes[1].legend()
    #
    # plt.show()

    testing = testing[testing["location"] == "United Kingdom"]

    testing["date"] = pd.to_datetime(testing["date"])

    fig, ax1 = plt.subplots(figsize=(10, 9))

    ax2 = ax1.twinx()

    ax1.plot(testing["date"], testing["new_cases_smoothed_per_million"], label="New Cases")

    ax1.plot(testing["date"], 30 * testing["new_deaths_smoothed_per_million"], label="New Deaths (scaled)")
    ax1.plot(testing["date"], 10 * testing["icu_patients_per_million"], label="ICU Patients per Million")
    ax1.plot(testing["date"], 300 * np.log(testing["reproduction_rate"]), label="R")
    ax1.grid()
    ax2.plot(testing["date"], testing["people_fully_vaccinated_per_hundred"], label="Vaccination Rate", color="green")

    plt.legend(*unify_legend(ax1, ax2))
    plt.show()


def test():

    n_policies = 82
    n_other = 1

    model = nn.Sequential(
        nn.Linear(in_features=n_policies + n_other, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = nn.MSELoss()

    schedulers = []

    train_loader = None
    val_loader = None

    policy_tracker = PolicyTracker(model, optimizer, loss, schedulers, train_loader, val_loader)

    policy_tracker.fit(100, verbosity=2)


def unify_legend(*axes):
    hs = []
    ls = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()

        hs += h
        ls += l

    return hs, ls


if __name__ == "__main__":
    main()
