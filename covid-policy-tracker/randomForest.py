import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from datamodule import ResponseDataModule
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.model_selection import train_test_split


def plot_country(model, df, country="Germany", randomize_policies=False):
    df = df[df["country"] == country]

    df.pop("country")

    y = df.pop("reproduction_rate").to_numpy()
    x = df.to_numpy()

    # drop index
    x = x[:, 1:]

    n_policies = 46

    if randomize_policies:
        # this is useful to check how much the model relies on this vs other features
        random_x = np.random.randint(0, 1, size=(x.shape[0], n_policies))
        x[:, :n_policies] = random_x

    predicted = model.predict(x)

    ax = plt.gca()
    ax.plot(y, label="Actual")
    ax.plot(predicted, label="Predicted")
    ax.set_xlabel("Time")
    ax.set_ylabel("R")
    ax.set_title(country)
    ax.legend()
    
def plot_countries(model, countries=("Germany",), randomize_policies=False, dataset=""):

    df = pd.read_csv(dataset + "policies_onehot_full_absolute_R.csv")

    nrows = int(round(np.sqrt(len(countries))))
    ncols = len(countries) // nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols + 1, 6 * nrows))

    for i, country in enumerate(countries):
        plt.subplot(nrows, ncols, i + 1)
        plot_country(model, df, country, randomize_policies=randomize_policies)

    # set all ylims equal
    ylims = []
    for ax in axes.flat:
        ylims.extend(ax.get_ylim())

    ylims = [min(ylims), max(ylims)]
    for ax in axes.flat:
        ax.set_ylim(ylims)

    plt.savefig("test.png")
    # plt.show()

def plot_single_policy(model):
    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols)
    n_policies = 46
    n_other = 2

    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)

        for j in range(6):
            policy = np.zeros(n_policies + n_other)
            policy[j] = 1

            x = np.tile(policy, (101, 1))
            x[:,-2] = 2*i * np.ones(len(x))
            x[:,-1] = np.linspace(0,1,101)

            y = model.predict(x)

            ax = plt.gca()
            ax.plot(np.linspace(0,1,101), y, label=j)
            #ax.set_xlabel("Vaccinations")
            #ax.set_ylabel("Delta R")
            ax.set_title(f"{2*i} days")
            #ax.legend()

    # plt.show()

def plot_policies_vaccination(model, vaccination, dataset=""):
    n_policies = 46
    
    policies = np.eye(n_policies)

    x = np.zeros((n_policies+1, n_policies+2))
    x[1:,:-2] = policies
    x[:,-1] = vaccination * np.ones(n_policies+1)

    y = model.predict(x)

    plt.figure(figsize=(19,12))
    plt.scatter(np.arange(n_policies+1), y)

    if dataset == "":
        xticks = np.array(["no","C1 1","C1 2","C1 3","C2 1","C2 2","C2 3","C3 1","C3 2","C4 1","C4 2","C4 3","C4 4","C5 1","C5 2","C6 1","C6 2","C6 3","C7 1","C7 2","C8 1","C8 2","C8 3","C8 4","E1 1","E1 2","E2 1","E2 2","H1 1","H1 2","H2 1","H2 2","H2 3","H3 1","H3 2","H6 1","H6 2","H6 3","H6 4","H7 1","H7 2","H7 3","H7 4","H7 5","H8 1","H8 2","H8 3",])
    else:
        xticks = ["no","ActivateCaseNotification","ActivateOrEstablishEmergencyResponse","ActivelyCommunicateWithHealthcareProfessionals1","ActivelyCommunicateWithManagers1","AdaptProceduresForPatientManagement","AirportHealthCheck","AirportRestriction","BorderHealthCheck","BorderRestriction","ClosureOfEducationalInstitutions","CordonSanitaire","CrisisManagementPlans","EducateAndActivelyCommunicateWithThePublic1","EnhanceDetectionSystem","EnhanceLaboratoryTestingCapacity","EnvironmentalCleaningAndDisinfection","IncreaseAvailabilityOfPpe","IncreaseHealthcareWorkforce","IncreaseInMedicalSuppliesAndEquipment","IncreaseIsolationAndQuarantineFacilities","IncreasePatientCapacity","IndividualMovementRestrictions","IsolationOfCases","MassGatheringCancellation","MeasuresForPublicTransport","MeasuresForSpecialPopulations","MeasuresToEnsureSecurityOfSupply","NationalLockdown","PersonalProtectiveMeasures","PoliceAndArmyInterventions","PortAndShipRestriction","ProvideInternationalHelp","PublicTransportRestriction","Quarantine","ReceiveInternationalHelp","RepurposeHospitals","Research","RestrictedTesting","ReturnOperationOfNationals","SmallGatheringCancellation","SpecialMeasuresForCertainEstablishments","Surveillance","TheGovernmentProvideAssistanceToVulnerablePopulations","TracingAndTracking","TravelAlertAndWarning","WorkSafetyProtocols",]
    print(len(xticks))
    plt.xticks(np.arange(n_policies+1), xticks, rotation='vertical')
    plt.tight_layout()
    # plt.show()
    
def plot_policies_vaccination_gbr(low, median, mean, high, vaccination, dataset=""):
    n_policies = 46
    
    policies = np.eye(n_policies)

    x = np.zeros((n_policies+1, n_policies+2))
    x[1:,:-2] = policies
    x[:,-1] = vaccination * np.ones(n_policies+1)

    y_low = low.predict(x)
    y_median = median.predict(x)
    y_mean = mean.predict(x)
    y_high = high.predict(x)

    plt.figure(figsize=(19,12))
    plt.scatter(np.arange(n_policies+1), y_low, label="low")
    plt.scatter(np.arange(n_policies+1), y_median, label="mid")
    plt.scatter(np.arange(n_policies+1), y_mean, label="mean")
    plt.scatter(np.arange(n_policies+1), y_high, label="high")

    if dataset == "":
        xticks = np.array(["no","C1 1","C1 2","C1 3","C2 1","C2 2","C2 3","C3 1","C3 2","C4 1","C4 2","C4 3","C4 4","C5 1","C5 2","C6 1","C6 2","C6 3","C7 1","C7 2","C8 1","C8 2","C8 3","C8 4","E1 1","E1 2","E2 1","E2 2","H1 1","H1 2","H2 1","H2 2","H2 3","H3 1","H3 2","H6 1","H6 2","H6 3","H6 4","H7 1","H7 2","H7 3","H7 4","H7 5","H8 1","H8 2","H8 3",])
    else:
        xticks = ["no","ActivateCaseNotification","ActivateOrEstablishEmergencyResponse","ActivelyCommunicateWithHealthcareProfessionals1","ActivelyCommunicateWithManagers1","AdaptProceduresForPatientManagement","AirportHealthCheck","AirportRestriction","BorderHealthCheck","BorderRestriction","ClosureOfEducationalInstitutions","CordonSanitaire","CrisisManagementPlans","EducateAndActivelyCommunicateWithThePublic1","EnhanceDetectionSystem","EnhanceLaboratoryTestingCapacity","EnvironmentalCleaningAndDisinfection","IncreaseAvailabilityOfPpe","IncreaseHealthcareWorkforce","IncreaseInMedicalSuppliesAndEquipment","IncreaseIsolationAndQuarantineFacilities","IncreasePatientCapacity","IndividualMovementRestrictions","IsolationOfCases","MassGatheringCancellation","MeasuresForPublicTransport","MeasuresForSpecialPopulations","MeasuresToEnsureSecurityOfSupply","NationalLockdown","PersonalProtectiveMeasures","PoliceAndArmyInterventions","PortAndShipRestriction","ProvideInternationalHelp","PublicTransportRestriction","Quarantine","ReceiveInternationalHelp","RepurposeHospitals","Research","RestrictedTesting","ReturnOperationOfNationals","SmallGatheringCancellation","SpecialMeasuresForCertainEstablishments","Surveillance","TheGovernmentProvideAssistanceToVulnerablePopulations","TracingAndTracking","TravelAlertAndWarning","WorkSafetyProtocols",]
    print(len(xticks))
    plt.xticks(np.arange(n_policies+1), xticks, rotation='vertical')
    plt.legend()
    plt.tight_layout()

    # plt.show()

    
def plot_policies_vaccination_gbr_std(low, mean, high, vaccination, dataset=""):
    n_policies = 46
    
    policies = np.eye(n_policies)

    x = np.zeros((n_policies+1, n_policies+2))
    x[1:,:-2] = policies
    x[:,-1] = vaccination * np.ones(n_policies+1)

    y_low = low.predict(x)
    y_mean = mean.predict(x)
    y_high = high.predict(x)

    stds = (y_high - y_low) / (2 * stats.norm.ppf(high.alpha))

    plt.figure(figsize=(19,12))
    plt.errorbar(np.arange(n_policies+1), y_mean, yerr=stds, fmt=".", label="low")

    if dataset == "":
        xticks = np.array(["no","C1 1","C1 2","C1 3","C2 1","C2 2","C2 3","C3 1","C3 2","C4 1","C4 2","C4 3","C4 4","C5 1","C5 2","C6 1","C6 2","C6 3","C7 1","C7 2","C8 1","C8 2","C8 3","C8 4","E1 1","E1 2","E2 1","E2 2","H1 1","H1 2","H2 1","H2 2","H2 3","H3 1","H3 2","H6 1","H6 2","H6 3","H6 4","H7 1","H7 2","H7 3","H7 4","H7 5","H8 1","H8 2","H8 3",])
    else:
        xticks = ["no","ActivateCaseNotification","ActivateOrEstablishEmergencyResponse","ActivelyCommunicateWithHealthcareProfessionals1","ActivelyCommunicateWithManagers1","AdaptProceduresForPatientManagement","AirportHealthCheck","AirportRestriction","BorderHealthCheck","BorderRestriction","ClosureOfEducationalInstitutions","CordonSanitaire","CrisisManagementPlans","EducateAndActivelyCommunicateWithThePublic1","EnhanceDetectionSystem","EnhanceLaboratoryTestingCapacity","EnvironmentalCleaningAndDisinfection","IncreaseAvailabilityOfPpe","IncreaseHealthcareWorkforce","IncreaseInMedicalSuppliesAndEquipment","IncreaseIsolationAndQuarantineFacilities","IncreasePatientCapacity","IndividualMovementRestrictions","IsolationOfCases","MassGatheringCancellation","MeasuresForPublicTransport","MeasuresForSpecialPopulations","MeasuresToEnsureSecurityOfSupply","NationalLockdown","PersonalProtectiveMeasures","PoliceAndArmyInterventions","PortAndShipRestriction","ProvideInternationalHelp","PublicTransportRestriction","Quarantine","ReceiveInternationalHelp","RepurposeHospitals","Research","RestrictedTesting","ReturnOperationOfNationals","SmallGatheringCancellation","SpecialMeasuresForCertainEstablishments","Surveillance","TheGovernmentProvideAssistanceToVulnerablePopulations","TracingAndTracking","TravelAlertAndWarning","WorkSafetyProtocols",]
    print(len(xticks))
    plt.xticks(np.arange(n_policies+1), xticks, rotation='vertical')
    plt.legend()
    plt.tight_layout()

    # plt.show()


def calculate_permutation_importance(X_train, X_test, Y_train, Y_test, dataset=""):
    K = 100
    n_policies = 46

    model = RandomForestRegressor(n_estimators=100, max_depth=30, verbose=0, n_jobs=-1)
    model.fit(X_train, Y_train)
    baseline_score = model.score(X_test, Y_test)

    importances = np.zeros(shape=(n_policies, K))

    for policy_index in range(n_policies):
        print(f"policy_index {policy_index}")
        shuffled_scores = np.zeros(K)
        for k in range(K):
            X_shuffled = X_test.copy()
            indices = np.arange(X_shuffled.shape[0])
            np.random.shuffle(indices)
            X_shuffled[:, policy_index] = X_shuffled[:, policy_index][indices]
            
            shuffled_scores[k] = model.score(X_shuffled, Y_test)

        importances[policy_index] = baseline_score - shuffled_scores

    print(baseline_score)
    print(np.mean(importances, axis=1))
    print(np.std(importances, axis=1))

    
    if dataset == "":
        xticks = np.array(["C1 1","C1 2","C1 3","C2 1","C2 2","C2 3","C3 1","C3 2","C4 1","C4 2","C4 3","C4 4","C5 1","C5 2","C6 1","C6 2","C6 3","C7 1","C7 2","C8 1","C8 2","C8 3","C8 4","E1 1","E1 2","E2 1","E2 2","H1 1","H1 2","H2 1","H2 2","H2 3","H3 1","H3 2","H6 1","H6 2","H6 3","H6 4","H7 1","H7 2","H7 3","H7 4","H7 5","H8 1","H8 2","H8 3",])
    else:
        xticks = np.array(["ActivateCaseNotification","ActivateOrEstablishEmergencyResponse","ActivelyCommunicateWithHealthcareProfessionals1","ActivelyCommunicateWithManagers1","AdaptProceduresForPatientManagement","AirportHealthCheck","AirportRestriction","BorderHealthCheck","BorderRestriction","ClosureOfEducationalInstitutions","CordonSanitaire","CrisisManagementPlans","EducateAndActivelyCommunicateWithThePublic1","EnhanceDetectionSystem","EnhanceLaboratoryTestingCapacity","EnvironmentalCleaningAndDisinfection","IncreaseAvailabilityOfPpe","IncreaseHealthcareWorkforce","IncreaseInMedicalSuppliesAndEquipment","IncreaseIsolationAndQuarantineFacilities","IncreasePatientCapacity","IndividualMovementRestrictions","IsolationOfCases","MassGatheringCancellation","MeasuresForPublicTransport","MeasuresForSpecialPopulations","MeasuresToEnsureSecurityOfSupply","NationalLockdown","PersonalProtectiveMeasures","PoliceAndArmyInterventions","PortAndShipRestriction","ProvideInternationalHelp","PublicTransportRestriction","Quarantine","ReceiveInternationalHelp","RepurposeHospitals","Research","RestrictedTesting","ReturnOperationOfNationals","SmallGatheringCancellation","SpecialMeasuresForCertainEstablishments","Surveillance","TheGovernmentProvideAssistanceToVulnerablePopulations","TracingAndTracking","TravelAlertAndWarning","WorkSafetyProtocols",])
    
    sorted_indices = np.argsort(np.mean(importances, axis=1))

    fig, ax = plt.subplots(figsize=(12,12))
    ax.boxplot(importances[sorted_indices,:].T, vert=False, labels=xticks[sorted_indices])

    # plt.show()


def knockout_evaluation(model, dataset=""):
    df = pd.read_csv(dataset + "policies_onehot_full_absolute_R.csv")
    df.pop("country")
    y = df.pop("reproduction_rate").to_numpy()[..., np.newaxis]
    x = df.to_numpy()
    # drop index
    x = x[:, 1:]

    n_policies = 46
    
    if dataset == "":
        yticks = np.array(["C1 1","C1 2","C1 3","C2 1","C2 2","C2 3","C3 1","C3 2","C4 1","C4 2","C4 3","C4 4","C5 1","C5 2","C6 1","C6 2","C6 3","C7 1","C7 2","C8 1","C8 2","C8 3","C8 4","E1 1","E1 2","E2 1","E2 2","H1 1","H1 2","H2 1","H2 2","H2 3","H3 1","H3 2","H6 1","H6 2","H6 3","H6 4","H7 1","H7 2","H7 3","H7 4","H7 5","H8 1","H8 2","H8 3",])
    else:
        yticks = np.array(["ActivateCaseNotification","ActivateOrEstablishEmergencyResponse","ActivelyCommunicateWithHealthcareProfessionals1","ActivelyCommunicateWithManagers1","AdaptProceduresForPatientManagement","AirportHealthCheck","AirportRestriction","BorderHealthCheck","BorderRestriction","ClosureOfEducationalInstitutions","CordonSanitaire","CrisisManagementPlans","EducateAndActivelyCommunicateWithThePublic1","EnhanceDetectionSystem","EnhanceLaboratoryTestingCapacity","EnvironmentalCleaningAndDisinfection","IncreaseAvailabilityOfPpe","IncreaseHealthcareWorkforce","IncreaseInMedicalSuppliesAndEquipment","IncreaseIsolationAndQuarantineFacilities","IncreasePatientCapacity","IndividualMovementRestrictions","IsolationOfCases","MassGatheringCancellation","MeasuresForPublicTransport","MeasuresForSpecialPopulations","MeasuresToEnsureSecurityOfSupply","NationalLockdown","PersonalProtectiveMeasures","PoliceAndArmyInterventions","PortAndShipRestriction","ProvideInternationalHelp","PublicTransportRestriction","Quarantine","ReceiveInternationalHelp","RepurposeHospitals","Research","RestrictedTesting","ReturnOperationOfNationals","SmallGatheringCancellation","SpecialMeasuresForCertainEstablishments","Surveillance","TheGovernmentProvideAssistanceToVulnerablePopulations","TracingAndTracking","TravelAlertAndWarning","WorkSafetyProtocols",])
    
    for policy_index in range(n_policies):
        mask = (x[:, policy_index] == 1)
        print(f"policy_index: {policy_index+1:2d}/{n_policies} -> {np.sum(mask):7d} instances\t({yticks[policy_index]})")


    means = np.zeros(shape=n_policies)
    stds = np.zeros(shape=n_policies)
    ci = np.zeros(shape=n_policies)
    confidence = 0.95

    for policy_index in range(n_policies):
        mask = (x[:, policy_index] == 1)
        print(f"\npolicy_index: {policy_index+1}/{n_policies} -> {np.sum(mask):7d} instances\t({yticks[policy_index]})")
        
        if np.sum(mask) == 0:
            continue

        diff = np.array([])
        
        features = x[mask,:]
        
        print("\nbase")
        base_predictions = model.predict(features)

        print("\nknockout")
        features[:, policy_index] = 0
        knockout_predictions = model.predict(features)

        diff = base_predictions - knockout_predictions

        print(diff.shape)
        diff = diff[~np.isnan(diff)]


        means[policy_index] = np.mean(diff)
        stds[policy_index] = np.std(diff)
        ci[policy_index] = stats.sem(diff) * stats.t.ppf((1 + confidence) / 2., len(diff)-1)

        print(f"{yticks[policy_index]}\t{means[policy_index]}\t{stds[policy_index]}\t{ci[policy_index]}")

    print(means)
    print(stds)
    print(ci)

    plt.figure(figsize=(12, 12))
    plt.errorbar(means, -np.arange(n_policies), xerr=ci, fmt='.')
    plt.axvline(x=0.0, color="b")
    plt.yticks(-np.arange(n_policies), yticks, rotation='horizontal')
    for i, tick in enumerate(plt.gca().get_yticklabels()):
        tick.set_color("green" if means[i] < 0 else "red")

    sorted_indices = np.argsort(means)
    plt.figure(figsize=(12, 12))
    plt.errorbar(means[sorted_indices], -np.arange(n_policies), xerr=ci[sorted_indices], fmt='.')
    plt.axvline(x=0.0, color="b")
    plt.yticks(-np.arange(n_policies), yticks[sorted_indices], rotation='horizontal')
    for i, tick in enumerate(plt.gca().get_yticklabels()):
        tick.set_color("green" if means[sorted_indices][i] < 0 else "red")
        

def knockout_evaluation_same_category(model, dataset=""):
    if dataset != "":
        print("only for OxCGRT dataset!")
        return

    df = pd.read_csv(dataset + "policies_onehot_full_absolute_R.csv")
    df.pop("country")
    y = df.pop("reproduction_rate").to_numpy()[..., np.newaxis]
    x = df.to_numpy()
    # drop index
    x = x[:, 1:]

    n_policies = 46

    same_category_index_difference = [
        [ 0, +1, +2],               # C1 1
        [-1,  0, +1],               # C1 2
        [-2, -1,  0],               # C1 3
        
        [ 0, +1, +2],               # C2 1
        [-1,  0, +1],               # C2 2
        [-2, -1,  0],               # C2 3

        [ 0, +1],                   # C3 1
        [-1,  0],                   # C3 2
        
        [ 0, +1, +2, +3],           # C4 1
        [-1,  0, +1, +2],           # C4 2
        [-2, -1,  0, +1],           # C4 3
        [-3, -2, -1,  0],           # C4 4

        [ 0, +1],                   # C5 1
        [-1,  0],                   # C5 2
        
        [ 0, +1, +2],               # C6 1
        [-1,  0, +1],               # C6 2
        [-2, -1,  0],               # C6 3

        [ 0, +1],                   # C7 1
        [-1,  0],                   # C7 2
        
        [ 0, +1, +2, +3],           # C8 1
        [-1,  0, +1, +2],           # C8 2
        [-2, -1,  0, +1],           # C8 3
        [-3, -2, -1,  0],           # C8 4

        [ 0, +1],                   # E1 1
        [-1,  0],                   # E1 2

        [ 0, +1],                   # E2 1
        [-1,  0],                   # E2 2

        [ 0, +1],                   # H1 1
        [-1,  0],                   # H1 2
        
        [ 0, +1, +2],               # H2 1
        [-1,  0, +1],               # H2 2
        [-2, -1,  0],               # H2 3

        [ 0, +1],                   # H3 1
        [-1,  0],                   # H3 2
        
        [ 0, +1, +2, +3],           # H6 1
        [-1,  0, +1, +2],           # H6 2
        [-2, -1,  0, +1],           # H6 3
        [-3, -2, -1,  0],           # H6 4
        
        [ 0, +1, +2, +3, +4],       # H7 1
        [-1,  0, +1, +2, +3],       # H7 2
        [-2, -1,  0, +1, +2],       # H7 3
        [-3, -2, -1,  0, +1],       # H7 4
        [-4, -3, -2, -1,  0],       # H7 5
        
        [ 0, +1, +2],               # H8 1
        [-1,  0, +1],               # H8 2
        [-2, -1,  0],               # H8 3
    ]
    yticks = np.array(["C1 1","C1 2","C1 3","C2 1","C2 2","C2 3","C3 1","C3 2","C4 1","C4 2","C4 3","C4 4","C5 1","C5 2","C6 1","C6 2","C6 3","C7 1","C7 2","C8 1","C8 2","C8 3","C8 4","E1 1","E1 2","E2 1","E2 2","H1 1","H1 2","H2 1","H2 2","H2 3","H3 1","H3 2","H6 1","H6 2","H6 3","H6 4","H7 1","H7 2","H7 3","H7 4","H7 5","H8 1","H8 2","H8 3",])

    means = []
    stds = []
    cis = []
    confidence = 0.95

    for policy_index, index_differences in enumerate(same_category_index_difference):
        means.append(np.zeros(1 + len(index_differences)))
        stds.append(np.zeros(1 + len(index_differences)))
        cis.append(np.zeros(1 + len(index_differences)))

        mask = (x[:, policy_index] == 1)
        # print(f"\npolicy_index: {policy_index+1}/{n_policies} -> {np.sum(mask):7d} instances\t({yticks[policy_index]})")
        
        if np.sum(mask) == 0:
            continue
            

        features = x[mask,:]
        # print("\nbase")
        base_predictions = model.predict(features)

        # print("\nknockout")
        features[:, policy_index] = 0
        knockout_predictions = model.predict(features)

        knockout_diff = base_predictions - knockout_predictions
        knockout_diff = knockout_diff[~np.isnan(knockout_diff)]
        means[-1][0] = np.mean(knockout_diff)
        stds[-1][0] = np.std(knockout_diff)
        cis[-1][0] = stats.sem(knockout_diff) * stats.t.ppf((1 + confidence) / 2., len(knockout_diff)-1)

        for i, active_index_diff in enumerate(index_differences):
            if active_index_diff == 0:
                continue

            # print(f"\nchanged policy ({yticks[policy_index+active_index_diff]})")
            # new policy from same category
            features[:, policy_index+active_index_diff] = 1

            knockout_predictions = model.predict(features)

            knockout_diff = base_predictions - knockout_predictions
            knockout_diff = knockout_diff[~np.isnan(knockout_diff)]
            means[-1][i+1] = np.mean(knockout_diff)
            stds[-1][i+1] = np.std(knockout_diff)
            cis[-1][i+1] = stats.sem(knockout_diff) * stats.t.ppf((1 + confidence) / 2., len(knockout_diff)-1)

            # reset
            features[:, policy_index+active_index_diff] = 0

        with np.printoptions(precision=4, suppress=True, linewidth=200):
            print(means[-1])



    plt.figure(figsize=(19,19))

    for policy_index, index_differences in enumerate(same_category_index_difference):
        ax = plt.subplot(6, 8, policy_index+1)
        
        ax.errorbar(means[policy_index], -np.arange(len(means[policy_index])), xerr=cis[policy_index], fmt=".")
        ax.axvline(x=0.0, color="black")

        ax.set_yticks(-np.arange(len(means[policy_index])))
        ax.set_yticklabels(["no"] + list(yticks[policy_index+index_differences[0]:policy_index+index_differences[-1]+1]))

        ax.set_title(yticks[policy_index])

    plt.subplots_adjust(hspace=1, wspace=.5)
    





def main():
    dataset = ""  # "CCCSL_" or ""
    df = pd.read_csv(dataset + "policies_onehot_full_absolute_R.csv")
    df = df.copy()
    df.pop("country")
    Y = df.pop("reproduction_rate").to_numpy()
    X = df.to_numpy()
    # drop index
    X = X[:, 1:]

    train_features, val_features, train_responses, val_responses = train_test_split(X, Y, test_size=0.2, random_state=42)

    calculate_permutation_importance(train_features, val_features, train_responses, val_responses, dataset=dataset)

    rf = RandomForestRegressor(n_estimators=100, max_depth=30, verbose=0, n_jobs=-1)
    rf.fit(train_features, train_responses)

    knockout_evaluation(rf, dataset=dataset)
    knockout_evaluation_same_category(rf, dataset=dataset)
    

    countries = ("Germany", "Spain", "Italy", "Japan", "New Zealand", "Ecuador")

    # plot_countries(rf, countries, randomize_policies=True, dataset=dataset)
    plot_countries(rf, countries, randomize_policies=False, dataset=dataset)

    # plot_single_policy(rf)
    # plot_policies_vaccination(rf, 0, dataset=dataset)
    # plot_policies_vaccination(rf, 1, dataset=dataset)


    # gbr_low = GradientBoostingRegressor(loss="quantile", alpha=0.1, verbose=1)
    # gbr_high = GradientBoostingRegressor(loss="quantile", alpha=0.9, verbose=1)
    # gbr_median = GradientBoostingRegressor(loss="quantile", alpha=0.5, verbose=1)
    # gbr_mean = GradientBoostingRegressor(loss="ls", verbose=1)
    # gbr_low.fit(train_features, train_responses)
    # gbr_high.fit(train_features, train_responses)
    # gbr_median.fit(train_features, train_responses)
    # gbr_mean.fit(train_features, train_responses)
    #plot_policies_vaccination_gbr(gbr_low, gbr_median, gbr_mean, gbr_high, 0, dataset=dataset)

    #plot_policies_vaccination_gbr_std(gbr_low, gbr_mean, gbr_high, 0, dataset=dataset)

    plt.show()
    

if __name__ == "__main__":
    main()
