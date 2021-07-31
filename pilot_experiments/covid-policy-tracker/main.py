
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from datetime import datetime, timedelta

import data
from policy_tracker import PolicyTracker
from trainer import trainer


def main():
    responses, responses2, testing = data.fetch()

    #responses2.reencode("cp1252", "utf8")

    #responses.save()
    #responses2.save()
    #testing.save()

    responses.load()
    responses2.load()
    testing.load()

    responses = responses.to_df(low_memory=False)
    responses2 = responses2.to_df()
    testing = testing.to_df()

    print(responses.dtypes)
    print("========================================")
    print(responses2.dtypes)
    print("========================================")
    print(testing.dtypes)

    print(responses.max())

    # convert measures to one-hot
    # lb = LabelBinarizer()
    # responses2["L2"] = lb.fit_transform(responses2["Measure_L2"]).tolist()
    # print(responses2["L2"])

    testing = testing[testing["location"] == "Germany"]

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
    pt = PolicyTracker()
    trainer.fit(pt)


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
    test()
