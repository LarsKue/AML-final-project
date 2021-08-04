

from data import GitHubData
import pandas as pd

import torch


def create_policies_oh(responses, testing, country_name):
    print(country_name)

    # filter for one country, for now
    responses_single = responses[responses["CountryName"] == country_name].copy()
    testing_sinlge = testing[testing["location"] == country_name].copy()

    # superset of dates from start to finish
    dates = pd.date_range(start=responses_single["Date"].min(), end=responses_single["Date"].max(), freq="1D")

    # set the new dates as index for both dataframes
    responses_single.set_index("Date", inplace=True)
    responses_single = responses_single.reindex(dates)

    testing_sinlge.set_index("date", inplace=True)
    testing_sinlge = testing_sinlge.reindex(dates)

    responses_single = responses_single.reset_index()
    testing_sinlge = testing_sinlge.reset_index()

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

    max_values = [3, 3, 2, 4, 2, 3, 2, 4, 2, 2, 2, 3, 2, 4, 5, 3]
    n = sum(max_values)

    policies_oh_single = torch.zeros(len(policies), n)

    for i, row in policies.iterrows():
        oh = torch.zeros(n)
        offset = 0

        for j, index in enumerate(row):
            try:
                oh[offset + int(index) - 1] = 1.0
            except ValueError:
                # nan
                pass

            offset += max_values[j]

        policies_oh_single[i] = oh

    policies_oh_single = pd.DataFrame(policies_oh_single.numpy())

    # relevant data from testing is in vaccinations and reproduction rate
    policies_oh_single["vaccination_rate"] = testing_sinlge["people_fully_vaccinated_per_hundred"].fillna(0.0)
    policies_oh_single["reproduction_rate"] = testing_sinlge["reproduction_rate"]

    policies_oh_single = policies_oh_single.dropna()

    return policies_oh_single


responses = GitHubData(user="OxCGRT", repo="covid-policy-tracker", branch="master", path="data/OxCGRT_latest.csv")
testing = GitHubData(user="owid", repo="covid-19-data", branch="master", path="public/data/owid-covid-data.csv")


responses.save()
testing.save()

responses.load()
testing.load()

responses = responses.to_df(low_memory=False)
testing = testing.to_df()

responses["Date"] = pd.to_datetime(responses["Date"], format="%Y%m%d")
testing["date"] = pd.to_datetime(testing["date"], format="%Y-%m-%d")

print(responses.dtypes)
print(testing.dtypes)

print(responses["CountryName"].unique())
print(testing["location"].unique())
intersection = [x for x in responses["CountryName"].unique() if x in testing["location"].unique() and x not in ["Germany", "Brazil", "Canada", "China", "United Kingdom", "United States"]]
print(intersection)
policies_oh = create_policies_oh(responses, testing, intersection[0])
print(policies_oh)

for country_name in intersection[1:]:
    policies_oh = policies_oh.append(create_policies_oh(responses, testing, country_name))

policies_oh.to_csv("policies_onehot_full.csv")

create_policies_oh(responses, testing, "Germany").to_csv("policies_onehot_Germany.csv")