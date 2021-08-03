

from data import GitHubData
import pandas as pd

import torch


responses = GitHubData(user="OxCGRT", repo="covid-policy-tracker", branch="master", path="data/OxCGRT_latest.csv")
testing = GitHubData(user="owid", repo="covid-19-data", branch="master", path="public/data/owid-covid-data.csv")

responses.load()
testing.load()

responses = responses.to_df(low_memory=False)
testing = testing.to_df()

responses["Date"] = pd.to_datetime(responses["Date"], format="%Y%m%d")
testing["date"] = pd.to_datetime(testing["date"], format="%Y-%m-%d")

print(responses.dtypes)
print(testing.dtypes)

# filter for one country, for now
responses = responses[responses["CountryName"] == "Germany"]
testing = testing[testing["location"] == "Germany"]

# superset of dates from start to finish
dates = pd.date_range(start=responses["Date"].min(), end=responses["Date"].max(), freq="1D")

# set the new dates as index for both dataframes
responses.set_index("Date", inplace=True)
responses = responses.reindex(dates)

testing.set_index("date", inplace=True)
testing = testing.reindex(dates)

responses = responses.reset_index()
testing = testing.reset_index()

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

policies = responses[policy_names]

max_values = [3, 3, 2, 4, 2, 3, 2, 4, 2, 2, 2, 3, 2, 4, 5, 3]
n = sum(max_values)

policies_oh = torch.zeros(len(policies), n)

for i, row in policies.iterrows():
    oh = torch.zeros(n)
    offset = 0

    for j, index in enumerate(row):
        oh[offset + int(index) - 1] = 1.0

        offset += max_values[j]

    policies_oh[i] = oh

policies_oh = pd.DataFrame(policies_oh.numpy())

# relevant data from testing is in vaccinations and reproduction rate
policies_oh["vaccination_rate"] = testing["people_fully_vaccinated_per_hundred"].fillna(0.0)
policies_oh["reproduction_rate"] = testing["reproduction_rate"]

policies_oh = policies_oh.dropna()

policies_oh.to_csv("policies_onehot.csv")
