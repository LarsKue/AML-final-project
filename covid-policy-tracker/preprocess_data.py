import math

import numpy as np
import pandas as pd
import torch

from scipy.signal import savgol_filter

from data import GitHubData


def window(l, n):
    for i in range(len(l) - n):
        yield l[i:i + n]


def policies_oh_from_indices(policies, max_values=(3, 3, 2, 4, 2, 3, 2, 4, 2, 2, 2, 3, 2, 4, 5, 3)):
    n = sum(max_values)
    policies_oh_single = torch.zeros(len(policies), n)

    days_since_application = torch.zeros(len(policies))
    previous_policies_indices = policies.iloc[0, :].to_numpy()

    for i, row in policies.iterrows():
        oh = torch.zeros(n)
        policies_nan_0 = np.zeros(n)
        offset = 0

        contains_nan = False
        for j, index in enumerate(row):
            try:
                if int(index) != 0:
                    policies_nan_0[offset + int(index) - 1] = 1.0
            except ValueError:
                policies_nan_0[j] = 0
                contains_nan = True
            offset += max_values[j]

        if np.array_equal(previous_policies_indices, policies_nan_0):
            days_since_application[i] = days_since_application[i - 1] + 1
        previous_policies_indices = policies_nan_0

        if contains_nan:
            torch.fill_(oh, math.nan)
        else:
            offset = 0
            for j, index in enumerate(row):
                if int(index) != 0:
                    oh[offset + int(index) - 1] = 1.0
                offset += max_values[j]

        policies_oh_single[i] = oh

    policies_oh_single = pd.DataFrame(policies_oh_single.numpy())

    policies_oh_single["days_since_application"] = pd.Series(days_since_application)

    return policies_oh_single


def create_policies_oh(responses, testing, country_name, single_policy):
    print(country_name)

    # filter for one country, for now
    responses_single = responses[responses["CountryName"] == country_name].copy()
    testing_single = testing[testing["location"] == country_name].copy()

    # superset of dates from start to finish
    dates = pd.date_range(start=responses_single["Date"].min(), end=responses_single["Date"].max(), freq="1D")

    # set the new dates as index for both dataframes
    responses_single.set_index("Date", inplace=True)
    responses_single = responses_single.reindex(dates)

    testing_single.set_index("date", inplace=True)
    testing_single = testing_single.reindex(dates)

    responses_single = responses_single.reset_index()
    testing_single = testing_single.reset_index()

    # TODO: Maybe don't interpolate, and instead just drop NaNs if they are only at start and end
    # # interpolate the nan values using a piecewise constant function
    # # TODO: linear interpolation is also feasible (method="linear")
    # testing = testing.interpolate(method="pad")

    policy_names = [
        "C1_School closing",
        "C2_Workplace closing",
        "C3_Cancel public events",
        "C4_Restrictions on gatherings",
        "C5_Close public transport",
        "C6_Stay at home requirements",
        "C7_Restrictions on internal movement",
        "C8_International travel controls",
        "E1_Income support",
        "E2_Debt/contract relief",
        "H1_Public information campaigns",
        "H2_Testing policy",
        "H3_Contact tracing",
        "H6_Facial Coverings",
        "H7_Vaccination policy",
        "H8_Protection of elderly people"
    ]

    policies = responses_single[policy_names]

    policies_oh_single = policies_oh_from_indices(policies)

    # relevant data from testing is in vaccinations and reproduction rate
    policies_oh_single["vaccination_rate"] = testing_single["people_fully_vaccinated_per_hundred"].fillna(0.0) / 100

    # reproduction_rate = testing_single["reproduction_rate"].to_list()
    # reproduction_rate.append(math.nan)
    # reproduction_rate = np.array(reproduction_rate)
    # difference = reproduction_rate[1:] - reproduction_rate[:-1]
    # mask = np.isnan(difference)
    # difference[mask] = 0
    # difference = savgol_filter(difference, 21, 3)
    # difference[mask] = np.nan

    # policies_oh_single["reproduction_rate"] = difference
    policies_oh_single["reproduction_rate"] = testing_single["reproduction_rate"]

    policies_oh_single["country"] = country_name
    policies_oh_single["dates"] = dates

    # TODO: remove r value?
    # missing = list(np.full(7, np.nan))
    # shifted_r = missing + testing_single["reproduction_rate"][7:].tolist()
    # shifted_r = pd.Series(shifted_r)
    # policies_oh_single["reproduction_rate_last_week"] = shifted_r

    # TODO: give the model a window of time to see the cases, instead of just cases from exactly one week ago
    # offset = 14
    # missing_fill = list(np.full((offset,), np.nan))
    # shifted_cases = missing_fill + testing_single["new_cases_smoothed_per_million"][offset:].tolist()
    # shifted_cases = pd.Series(shifted_cases)
    # policies_oh_single[f"cases_d-{offset}"] = shifted_cases

    #
    # # TODO: Remove, or prevent overfit to this
    # cases_window = missing + list(window(testing_single["new_cases_smoothed_per_million"].tolist(), 7))[14:]
    # cases_window = np.array(cases_window)
    #
    # for i in range(7):
    #     s = pd.Series(cases_window[:, i])
    #     policies_oh_single[f"cases_window_{i}"] = s

    policies_oh_single = policies_oh_single.dropna()

    if single_policy and len(policies_oh_single) > 0:

        columns = policies_oh_single.columns

        days_since_application = -1 * np.ones(len(policy_names))
        previous_policies_indices = policies.loc[policies_oh_single.iloc[0, :].head(0).name].to_numpy()

        policies_oh_single_policy = pd.DataFrame(columns=policies_oh_single.columns)
        for i in range(len(policies_oh_single)):
            row = policies_oh_single.iloc[i, :]
            policies_oh_row = row[:n].to_numpy()

            policies_indices = policies.loc[row.head(0).name].to_numpy()
            same_policies = (policies_indices - previous_policies_indices) == 0
            days_since_application[same_policies] += 1
            days_since_application[~same_policies] = 0

            previous_policies_indices = policies_indices

            if policies_oh_row.sum() != 0:
                policies_oh_row = np.diag(policies_oh_row)
                policies_oh_row = policies_oh_row[~np.all(policies_oh_row == 0, axis=1)]

            df = pd.DataFrame(policies_oh_row)

            for c in columns[n:]:
                df[c] = row[c]

            df["days_since_application"] = pd.Series(days_since_application[policies_indices != 0])

            policies_oh_single_policy = policies_oh_single_policy.append(df)

        return policies_oh_single_policy

    else:
        return policies_oh_single


responses = GitHubData(user="OxCGRT", repo="covid-policy-tracker", branch="master", path="data/OxCGRT_latest.csv")
testing = GitHubData(user="owid", repo="covid-19-data", branch="master", path="public/data/owid-covid-data.csv")

# responses.save()
# testing.save()

responses.load()
testing.load()

responses = responses.to_df(low_memory=False)
testing = testing.to_df()

responses["Date"] = pd.to_datetime(responses["Date"], format="%Y%m%d")
testing["date"] = pd.to_datetime(testing["date"], format="%Y-%m-%d")

# create_policies_oh(responses, testing, "Germany", False).to_csv("policies_onehot_Germany_single_policy.csv")
# exit()

# print(responses.dtypes)
# print(testing.dtypes)

# print(responses["CountryName"].unique())
# print(testing["location"].unique())


exclude = ["Germany", "Brazil", "Canada", "China", "United Kingdom", "United States"]

intersection = [x for x in responses["CountryName"].unique() if
                x in testing["location"].unique() and x not in exclude]
# print(intersection)

policies_oh_single_policy = create_policies_oh(responses, testing, intersection[0], False)
for country_name in intersection[1:]:
    policies_oh_single_policy = policies_oh_single_policy.append(
        create_policies_oh(responses, testing, country_name, False))
policies_oh_single_policy.to_csv("policies_onehot_full_wo_Germany_absolute_R.csv")

policies_oh_full = create_policies_oh(responses, testing, intersection[0], False)
for country_name in intersection[1:]:
    policies_oh_full = policies_oh_full.append(create_policies_oh(responses, testing, country_name, False))
policies_oh_full = policies_oh_full.append(create_policies_oh(responses, testing, "Germany", False))
policies_oh_full.to_csv("policies_onehot_full_absolute_R.csv")

create_policies_oh(responses, testing, "Germany", False).to_csv("policies_onehot_Germany_absolute_R.csv")
