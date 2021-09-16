import math
import datetime

import numpy as np
import pandas as pd
import torch

from scipy.signal import savgol_filter

from data import GitHubData

def create_policies_oh(features, responses, country_name, single_policy):
    print(country_name)

    # filter for one country, for now
    features_single = features[features["Country"] == country_name].copy()
    responses_single = responses[responses["location"] == country_name].copy()

    # superset of dates from start to finish
    dates = pd.date_range(start=max(features_single["Date"].min(), responses_single["date"].min()), 
                            end=min(features_single["Date"].max(), responses_single["date"].max()), 
                            freq="1D")

    # set the new dates as index for both dataframes
    features_single.set_index("Date", inplace=True)
    features_single = features_single.reindex(dates)

    responses_single.set_index("date", inplace=True)
    responses_single = responses_single.reindex(dates)

    features_single = features_single.reset_index()
    responses_single = responses_single.reset_index()


    policies_oh_single = (features_single.iloc[:, 12:-5] * 1).copy()

    policies_numpy = policies_oh_single.to_numpy()

    days_since_application_vector = np.zeros(policies_numpy.shape[0])
    days_since_application = -1
    previous_policies = np.ones(policies_numpy.shape[1]) * -1
    for i in range(policies_numpy.shape[0]):
        if np.array_equal(policies_numpy[i], previous_policies):
            days_since_application += 1
            days_since_application_vector[i] = days_since_application
        else:
            previous_policies = policies_numpy[i]
            days_since_application = 0
            days_since_application_vector[i] = days_since_application

    policies_oh_single["days_since_application"] = pd.Series(days_since_application_vector)
    
    # relevant data from testing is in vaccinations and reproduction rate
    policies_oh_single["vaccination_rate"] = responses_single["people_fully_vaccinated_per_hundred"].fillna(0.0)

    reproduction_rate = responses_single["reproduction_rate"].to_list()
    reproduction_rate.append(math.nan)
    reproduction_rate = np.array(reproduction_rate)
    difference = reproduction_rate[1:] - reproduction_rate[:-1]
    mask = np.isnan(difference)
    difference[mask] = 0
    difference = savgol_filter(difference, 21, 3)
    difference[mask] = np.nan

    policies_oh_single["reproduction_rate"] = difference
    # policies_oh_single["reproduction_rate"] = responses_single["reproduction_rate"]

    policies_oh_single["country"] = country_name

    policies_oh_single = policies_oh_single.dropna()

    return policies_oh_single


features = GitHubData(user="complexity-science-hub", repo="ranking_npis", branch="master", path="data/COVID19_data_cumulative_PAPER_VERSION.csv")
responses = GitHubData(user="owid", repo="covid-19-data", branch="master", path="public/data/owid-covid-data.csv")

features.save()
responses.save()

features.load()
responses.load()

features = features.to_df(sep=';')
responses = responses.to_df(low_memory=False)

#print(features.columns)

features["Date"] = pd.to_datetime(features["Date"], format="%Y-%m-%d")
responses["date"] = pd.to_datetime(responses["date"], format="%Y-%m-%d")


exclude = ["Germany", "Brazil", "Canada", "China", "United Kingdom", "United States"]

intersection = [x for x in features["Country"].unique() if
                x in responses["location"].unique() and 
                x not in exclude]

policies_oh_single_policy = create_policies_oh(features, responses, intersection[0], False)
for country_name in intersection[1:]:
    policies_oh_single_policy = policies_oh_single_policy.append(
        create_policies_oh(features, responses, country_name, False))
policies_oh_single_policy.to_csv("CCCSL_policies_onehot_full_wo_Germany.csv")

policies_oh_full = create_policies_oh(features, responses, intersection[0], False)
for country_name in intersection[1:]:
    policies_oh_full = policies_oh_full.append(create_policies_oh(features, responses, country_name, False))
policies_oh_full = policies_oh_full.append(create_policies_oh(features, responses, "Germany", False))
policies_oh_full.to_csv("CCCSL_policies_onehot_full.csv")

create_policies_oh(features, responses, "Germany", False).to_csv("CCCSL_policies_onehot_Germany.csv")
