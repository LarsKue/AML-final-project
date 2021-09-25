import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

yticks = np.array([
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

confidence = 0.95
n_policies = 46


models = []
for i, model_name in enumerate(["MDN", "ProductAnsatz"]):
    importances = np.load(f"permutation_importances_{model_name}.npy").T
    means = np.mean(importances, axis=0)
    ci = np.std(importances, axis=0) * stats.t.ppf((1 + confidence) / 2., len(importances)-1)
    models.append([means, ci])
    print(means.shape, ci.shape)

allFGPR = np.load(f"allFGPR_permutation_importances.npy").T
for i in [7, 3, 6]:
    importances = allFGPR[:,:,i]
    means = np.mean(importances, axis=0)
    ci = np.std(importances, axis=0) * stats.t.ppf((1 + confidence) / 2., len(importances)-1)
    models.append([means, ci])
    print(means.shape, ci.shape)

model_titles = ["Mixture Density Network", "Product Ansatz", "Distributed Gaussian Process - grBCM", "Distributed Gaussian Process - GPoE", "Distributed Gaussian Process - rBCM"]
model_names = ["MDN", "ProductAnsatz", "grBCM", "GPoE", "rBCM"]

sorted_indices_mom = []
for i, (means, ci) in enumerate(models):
    if i > 2:
        continue
    sorted_indices_mom.append(means)
sorted_indices_mom = np.argsort(np.mean(sorted_indices_mom, axis=0))
print(sorted_indices_mom.shape)


for i, (title, model_name, (means, ci)) in enumerate(zip(model_titles, model_names, models)):
    sorted_indices = np.argsort(means)

    plt.figure(figsize=(12, 12))
    plt.errorbar(means[sorted_indices]/np.sum(means), np.arange(n_policies), xerr=ci[sorted_indices]/np.sum(means), fmt='.', color=f"C{i+1}", capsize=5)
    plt.yticks(np.arange(n_policies), yticks[sorted_indices], rotation='horizontal')
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{model_name}_permutation_importance.png")
    plt.clf()
    plt.close()



plt.figure(figsize=(12,10))
pos = 0
for num_policies_per_category in [3, 3, 2, 4, 2, 3, 2, 4, 2, 2, 2, 3, 2, 4, 5, 3][:-1]:
    pos += num_policies_per_category
    plt.axhline(y=-(pos-.5), c="gray")

for i, (title, model_name, (means, ci)) in enumerate(zip(model_titles, model_names, models)):
    if i > 2:
        continue
    plt.errorbar(means/np.sum(means), -np.arange(n_policies), xerr=ci/np.sum(means), fmt='.', label=model_name, color=f"C{i+1}", capsize=5)

plt.yticks(-np.arange(n_policies), yticks, rotation='horizontal')
plt.xlabel("Feature Importance")
plt.title("Combined Feature Importances")
plt.tight_layout()
plt.legend()
plt.savefig("permutation_importance_combined_unsorted.png", format="png")
plt.clf()
plt.close()



plt.figure(figsize=(12,10))
for i, (title, model_name, (means, ci)) in enumerate(zip(model_titles, model_names, models)):
    if i > 2:
        continue
    plt.errorbar(means[sorted_indices_mom]/np.sum(means), np.arange(n_policies), xerr=ci[sorted_indices_mom]/np.sum(means), fmt='.', label=model_name, color=f"C{i+1}", capsize=5)

plt.yticks(np.arange(n_policies), yticks[sorted_indices_mom], rotation='horizontal')
plt.xlabel("Feature Importance")
plt.title("Combined Feature Importances")
plt.tight_layout()
plt.legend()
plt.legend()
plt.savefig("permutation_importance_combined_sorted_meanOfMeans.png", format="png")

