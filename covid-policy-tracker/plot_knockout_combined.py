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
    
n_policies = 46

MDN_means = np.load("MDN_knockout_means.npy")
MDN_ci = np.load("MDN_knockout_ci.npy")

PA_means = np.load("ProductAnsatz_knockout_means.npy")
PA_ci = np.load("ProductAnsatz_knockout_ci.npy")

grBCM_means = np.load("grBCM_knockout_means.npy")
grBCM_ci = np.load("grBCM_knockout_ci.npy")

allFGPR_means = np.load("allFGPR_knockout_means.npy")
allFGPR_ci = np.load("allFGPR_knockout_ci.npy")
# all = (mean, SPV, PoE, GPoE, GPoE_constant_beta, BCM, rBCM, grBCM)

models = [[MDN_means, MDN_ci], [PA_means, PA_ci], [allFGPR_means[7], allFGPR_ci[7]], [allFGPR_means[3], allFGPR_ci[3]], [allFGPR_means[6], allFGPR_ci[6]]]
for title, model_name, (means, ci) in zip(["Mixture Density Network", "Product Ansatz", "Distributed Gaussian Process - grBCM", "Distributed Gaussian Process - GPoE", "Distributed Gaussian Process - rBCM"], ["MDN", "ProductAnsatz", "grBCM", "GPoE", "rBCM"], models):
    sorted_indices = np.argsort(means)
    plt.figure(figsize=(12, 12))
    colors = np.where(means[sorted_indices] < 0, "green", "red")
    plt.errorbar(means[sorted_indices], -np.arange(n_policies), xerr=ci[sorted_indices], fmt='.', ecolor=colors)
    plt.axvline(x=0.0, color="black")
    plt.yticks(-np.arange(n_policies), yticks[sorted_indices], rotation='horizontal')
    plt.xlabel(r"$\Delta R_t$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{model_name}_knockout.png")
    plt.clf()
    plt.close()


mean_of_means = np.mean(np.array([mean for (mean, ci) in models]), axis=0)
mins = np.array([mean - ci for (mean, ci) in models])
maxs = np.array([mean + ci for (mean, ci) in models])
# mins = np.array([MDN_means - MDN_ci, PA_means - PA_ci, grBCM_means - grBCM_ci])
# maxs = np.array([MDN_means + MDN_ci, PA_means + PA_ci, grBCM_means + grBCM_ci])

mins = np.min(mins[:-2], axis=0)
maxs = np.max(maxs[:-2], axis=0)


plt.figure(figsize=(12, 12))
pos = 0
for num_policies_per_category in [3, 3, 2, 4, 2, 3, 2, 4, 2, 2, 2, 3, 2, 4, 5, 3][:-1]:
    pos += num_policies_per_category
    plt.axhline(y=-(pos-.5), c="gray")
    
for n in range(n_policies):
    plt.plot([mins[n], maxs[n]], [-n, -n], c="C0", zorder=0)
# plt.scatter(MDN_means, -np.arange(n_policies), c="C1", label="MDN", zorder=1)
# plt.scatter(PA_means, -np.arange(n_policies), c="C2", label="Product Ansatz", zorder=1)
# plt.scatter(grBCM_means, -np.arange(n_policies), c="C3", label="grBCM", zorder=1)
for i, (label, (mean, ci)) in enumerate(zip(["MDN", "ProductAnsatz", "grBCM", "GPoE", "rBCM"], models)):
    if i > 2:
        continue
    plt.scatter(mean, -np.arange(n_policies), c=f"C{i+1}", label=label, zorder=1)

plt.axvline(x=0.0, color="black", zorder=-1)
plt.yticks(-np.arange(n_policies), yticks, rotation='horizontal')
plt.xlabel(r"$\Delta R_t$")
plt.legend()
plt.tight_layout()
plt.savefig("Knockout_combined.png")


sorted_indices = np.argsort((maxs-mins)/2+mins)
sorted_indices = np.argsort(mean_of_means)
mins = mins[sorted_indices]
maxs = maxs[sorted_indices]

plt.figure(figsize=(12, 12))
for n in range(n_policies):
    plt.plot([mins[n], maxs[n]], [-n, -n], c="C0", zorder=0)
# plt.scatter(MDN_means[sorted_indices], -np.arange(n_policies), c="C1", label="MDN", zorder=1)
# plt.scatter(PA_means[sorted_indices], -np.arange(n_policies), c="C2", label="Product Ansatz", zorder=1)
# plt.scatter(grBCM_means[sorted_indices], -np.arange(n_policies), c="C3", label="grBCM", zorder=1)
for i, (label, (mean, ci)) in enumerate(zip(["MDN", "ProductAnsatz", "grBCM", "GPoE", "rBCM"], models)):
    if i > 2:
        continue
    plt.scatter(mean[sorted_indices], -np.arange(n_policies), c=f"C{i+1}", label=label, zorder=1)
plt.axvline(x=0.0, color="black", zorder=-1)
plt.yticks(-np.arange(n_policies), yticks[sorted_indices], rotation='horizontal')
plt.xlabel(r"$\Delta R_t$")
plt.legend()
plt.tight_layout()
plt.savefig("Knockout_combined_sorted_MeanOfMeans.png")