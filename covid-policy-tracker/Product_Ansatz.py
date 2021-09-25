import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import ResponseDataModule

import math
import subprocess as sub
import pathlib
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import pandas as pd
import datetime as dt


class PolicyTrackerProduct(pl.LightningModule):
    def __init__(self):
        super(PolicyTrackerProduct, self).__init__()

        # placeholder mean R0 taken from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7751056/
        # could be country-dependent, this should definitely be improved
        self.R0 = 3.28

        self.n_policies = 46
        self.n_other = 2

        self.model = nn.Sequential(
            nn.Linear(self.n_policies + self.n_other, 1024),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.n_policies),
        )

        # initialize weights
        for module in self.model.modules():
            if type(module) is nn.Linear:
                data = module.weight.data
                nn.init.normal_(data, 0, 2 / data.numel())

        # self.elementwiseProduct = ElementwiseProductLayer(self.n_policies)
        # nn.init.normal_(self.elementwiseProduct.weights.data, 1, 1)

        self.R0_model = nn.Linear(1, 1, bias=False)


        self.loss = nn.MSELoss()

    def forward(self, x, mc_dropout_samples=None):
        if mc_dropout_samples is None:
            # predict the effect p of single policies on the R value
            p = self.model(x)
            # p = self.elementwiseProduct(x[:,:-2])

            # take the product
            m = torch.prod(torch.exp(-p), dim=-1).unsqueeze(dim=-1)

            # scale with R0
            # return self.R0 * m
            res = self.R0_model(m)
            return res
        else:
            is_training = self.training

            self.train()
            samples = torch.zeros(size=(x.shape[0], mc_dropout_samples))
            for i in range(mc_dropout_samples):
                # if i % 100 == 0:
                #     print(f"sample nr. {i}")
                samples[:, i] = self(x).detach().squeeze()

            means = torch.mean(samples, axis=1)
            variances = torch.std(samples, axis=1)**2

            self.training = is_training
            return means, variances
            
    def score(self, X_test, Y_test, mc_dropout_samples=1000):
        means, _ = self(X_test, mc_dropout_samples=mc_dropout_samples)
        return 1 - (((Y_test.squeeze() - means)**2).sum() / ((Y_test.squeeze() - Y_test.squeeze().mean())**2).sum())
            
            
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.3, threshold=0.05,
                                                                  verbose=True)
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor="val_loss")

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)

        self.log("val_loss", loss)

        return loss




def tensorboard():
    """
    Create a detached process for tensorboard
    """
    args = ["tensorboard", "--logdir", "lightning_logs"]

    process = sub.Popen(
        args, shell=False, stdin=None, stdout=None, stderr=None,
        close_fds=True
    )

    return process


def latest_version(path):
    """
    Returns latest model version as integer
    """
    # unsorted
    versions = list(str(v.stem) for v in path.glob("version_*"))
    # get version numbers as integer
    versions = [re.match(r"version_(\d+)", version) for version in versions]
    versions = [int(match.group(1)) for match in versions]

    return max(versions)


def latest_checkpoint(version=None):
    """
    Returns latest checkpoint path for given version (default: latest) as string
    """
    path = pathlib.Path("lightning_logs")
    if version is None:
        version = latest_version(path)
    path = path / pathlib.Path(f"version_{version}/checkpoints/")

    checkpoints = list(str(cp.stem) for cp in path.glob("*.ckpt"))

    # find epoch and step numbers
    checkpoints = [re.match(r"epoch=(\d+)-step=(\d+)", cp) for cp in checkpoints]

    # assign steps to epochs
    epoch_steps = {}
    for match in checkpoints:
        epoch = match.group(1)
        step = match.group(2)

        if epoch not in epoch_steps:
            epoch_steps[epoch] = []

        epoch_steps[epoch].append(step)

    # find highest epoch and step
    max_epoch = max(epoch_steps.keys())
    max_step = max(epoch_steps[max_epoch])

    # reconstruct path
    checkpoint = path / f"epoch={max_epoch}-step={max_step}.ckpt"

    return str(checkpoint)


def plot_country(model, df, country="Germany", randomize_policies=False):
    df = df[df["country"] == country]

    df.pop("country")
    dates = df.pop("dates").to_numpy()
    dates_list = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
    first_day_indices = [i for (i, d) in enumerate(dates_list) if d.day == 1][::2]
    dates_list = [dates_list[i].strftime("%m/%Y") for i in first_day_indices]
    dates_list = [d[:3]+d[-2:] for d in dates_list]

    y = df.pop("reproduction_rate").to_numpy()
    x = df.to_numpy()

    # drop index
    x = x[:, 1:]

    x = torch.Tensor(x).cuda()
    y = torch.Tensor(y).unsqueeze(-1)

    if randomize_policies:
        # this is useful to check how much the model relies on this vs other features
        random_x = torch.randint(0, 1, size=(x.shape[0], model.n_policies)).to(torch.float64)
        x[:, :model.n_policies] = random_x

    # predicted = model(x).detach().cpu().numpy()
    means, variances = model(x, mc_dropout_samples=1000)

    means = means.detach().cpu().numpy()
    stds = torch.sqrt(variances).detach().cpu().numpy()

    ax = plt.gca()
    ax.plot(np.arange(len(y)), y, label="Actual")

    ax.plot(np.arange(len(y)), means, label="Predicted")
    ax.fill_between(np.arange(len(y)), means - 1.96*stds, means + 1.96*stds, color="C0", alpha=0.2, label="95% confidence")

    ax.set_xticks(first_day_indices)
    ax.set_xticklabels(dates_list, rotation=45)

    ax.set_ylabel("R")
    ax.set_title(country)
    ax.legend()
    print(ax.get_ylim())


def plot_countries(model, path, countries=("Germany",), randomize_policies=False, dataset=""):
    df = pd.read_csv(dataset + "policies_onehot_full_absolute_R.csv")

    nrows = int(round(np.sqrt(len(countries))))
    ncols = len(countries) // nrows

    plt.figure(figsize=(6 * ncols + 1, 6 * nrows))

    axes = []
    for i, country in enumerate(countries):
        axes.append(plt.subplot(nrows, ncols, i + 1))
        plot_country(model, df, country, randomize_policies=randomize_policies)

    # set all ylims equal
    ylims = []
    for ax in axes:
        ylims.extend(ax.get_ylim())

    ylims = [0, 3.75]
    for ax in axes:
        ax.set_ylim(ylims)

    plt.savefig(path + "countries_mc_dropout.png")
    plt.clf()
    plt.close()


def knockout_evaluation(model, path, vaccination_rate="any", dataset=""):
    df = pd.read_csv(dataset + "policies_onehot_full_absolute_R.csv")
    df.pop("country")
    dates = df.pop("dates")
    y = df.pop("reproduction_rate").to_numpy()[..., np.newaxis]
    x = df.to_numpy()
    # drop index
    x = torch.Tensor(x[:, 1:]).cuda()

    if vaccination_rate == "zero":
        vaccination_rate_mask = df["vaccination_rate"].to_numpy()
        vaccination_rate_mask = vaccination_rate_mask == 0
        x = x[vaccination_rate_mask,:]
        y = y[vaccination_rate_mask,:]
    elif vaccination_rate == "nonzero":
        vaccination_rate_mask = df["vaccination_rate"].to_numpy()
        vaccination_rate_mask = vaccination_rate_mask > 0
        x = x[vaccination_rate_mask,:]
        y = y[vaccination_rate_mask,:]


    n_policies = 46
    
    if dataset == "":
        yticks = np.array(["C1 1","C1 2","C1 3","C2 1","C2 2","C2 3","C3 1","C3 2","C4 1","C4 2","C4 3","C4 4","C5 1","C5 2","C6 1","C6 2","C6 3","C7 1","C7 2","C8 1","C8 2","C8 3","C8 4","E1 1","E1 2","E2 1","E2 2","H1 1","H1 2","H2 1","H2 2","H2 3","H3 1","H3 2","H6 1","H6 2","H6 3","H6 4","H7 1","H7 2","H7 3","H7 4","H7 5","H8 1","H8 2","H8 3",])
    else:
        yticks = np.array(["ActivateCaseNotification","ActivateOrEstablishEmergencyResponse","ActivelyCommunicateWithHealthcareProfessionals1","ActivelyCommunicateWithManagers1","AdaptProceduresForPatientManagement","AirportHealthCheck","AirportRestriction","BorderHealthCheck","BorderRestriction","ClosureOfEducationalInstitutions","CordonSanitaire","CrisisManagementPlans","EducateAndActivelyCommunicateWithThePublic1","EnhanceDetectionSystem","EnhanceLaboratoryTestingCapacity","EnvironmentalCleaningAndDisinfection","IncreaseAvailabilityOfPpe","IncreaseHealthcareWorkforce","IncreaseInMedicalSuppliesAndEquipment","IncreaseIsolationAndQuarantineFacilities","IncreasePatientCapacity","IndividualMovementRestrictions","IsolationOfCases","MassGatheringCancellation","MeasuresForPublicTransport","MeasuresForSpecialPopulations","MeasuresToEnsureSecurityOfSupply","NationalLockdown","PersonalProtectiveMeasures","PoliceAndArmyInterventions","PortAndShipRestriction","ProvideInternationalHelp","PublicTransportRestriction","Quarantine","ReceiveInternationalHelp","RepurposeHospitals","Research","RestrictedTesting","ReturnOperationOfNationals","SmallGatheringCancellation","SpecialMeasuresForCertainEstablishments","Surveillance","TheGovernmentProvideAssistanceToVulnerablePopulations","TracingAndTracking","TravelAlertAndWarning","WorkSafetyProtocols",])
    
    for policy_index in range(n_policies):
        mask = (x[:, policy_index] == 1)
        print(f"policy_index: {policy_index+1:2d}/{n_policies} -> {torch.sum(mask):7d} instances\t({yticks[policy_index]})")


    means = np.zeros(shape=n_policies)
    # stds = np.zeros(shape=n_policies)
    ci = np.zeros(shape=n_policies)
    confidence = 0.95

    for policy_index in range(n_policies):
        mask = (x[:, policy_index] == 1)
        # print(f"\npolicy_index: {policy_index+1}/{n_policies} -> {torch.sum(mask):7d} instances\t({yticks[policy_index]})")
        
        if torch.sum(mask) == 0:
            continue

        diff = np.array([])
        
        features = x[mask,:]
        
        # print("\nbase")
        base_predictions, base_variances = model(features, mc_dropout_samples=1000)

        # print("\nknockout")
        features[:, policy_index] = 0
        knockout_predictions, knockout_variances = model(features, mc_dropout_samples=1000)

        diff_means = (base_predictions - knockout_predictions).detach().cpu().numpy()
        diff_variances = (base_variances + knockout_variances).detach().cpu().numpy()

        print(diff_means.shape)
        mask = ~(np.isnan(diff_means) | np.isnan(diff_variances))
        diff_means = diff_means[mask]
        diff_variances = diff_variances[mask]

        means[policy_index] = np.mean(diff_means)
        # stds[policy_index] = np.std(diff_means)

        variance = np.mean(diff_means**2 + diff_variances) - np.mean(diff_means)**2
        sem = np.sqrt(variance) / np.sqrt(len(diff_means))
        ci[policy_index] = sem * stats.t.ppf((1 + confidence) / 2., len(diff_means)-1)

        print(f"{yticks[policy_index]}\t{means[policy_index]}\t{ci[policy_index]}")

        # means[policy_index] = np.mean(diff_means)
        # stds[policy_index] = np.std(diff_means)
        # ci[policy_index] = stats.sem(diff_means) * stats.t.ppf((1 + confidence) / 2., len(diff_means)-1)

        # print(f"{yticks[policy_index]}\t{means[policy_index]}\t{stds[policy_index]}\t{ci[policy_index]}")

    print(means)
    # print(stds)
    print(ci)
    np.save(path + f"knockout_means_vaccination_{vaccination_rate}.npy", means)
    # np.save(path + f"knockout_stds_vaccination_{vaccination_rate}.npy", stds)
    np.save(path + f"knockout_ci_vaccination_{vaccination_rate}.npy", ci)
    

    # plt.figure(figsize=(12, 12))
    # plt.errorbar(means, -np.arange(n_policies), xerr=ci, fmt='.')
    # plt.axvline(x=0.0, color="b")
    # plt.yticks(-np.arange(n_policies), yticks, rotation='horizontal')
    # for i, tick in enumerate(plt.gca().get_yticklabels()):
    #     tick.set_color("green" if means[i] < 0 else "red")

    sorted_indices = np.argsort(means)
    plt.figure(figsize=(12, 12))
    plt.errorbar(means[sorted_indices], -np.arange(n_policies), xerr=ci[sorted_indices], fmt='.')
    plt.axvline(x=0.0, color="black")
    plt.yticks(-np.arange(n_policies), yticks[sorted_indices], rotation='horizontal')
    for i, tick in enumerate(plt.gca().get_yticklabels()):
        tick.set_color("green" if means[sorted_indices][i] < 0 else "red")
    plt.xlabel(r"$\Delta R_t$")
    plt.title("Product Ansatz")
    plt.savefig(path + f"knockout_vaccination_{vaccination_rate}.png")
    plt.clf()
    plt.close()
        

def knockout_evaluation_same_category(model, dataset=""):
    if dataset != "":
        print("only for OxCGRT dataset!")
        return

    df = pd.read_csv(dataset + "policies_onehot_full_absolute_R.csv")
    df.pop("country")
    y = df.pop("reproduction_rate").to_numpy()[..., np.newaxis]
    x = df.to_numpy()
    # drop index
    x = torch.Tensor(x[:, 1:])

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
        # print(f"\npolicy_index: {policy_index+1}/{n_policies} -> {torch.sum(mask):7d} instances\t({yticks[policy_index]})")
        
        if torch.sum(mask) == 0:
            continue
            

        features = x[mask,:]
        # print("\nbase")
        base_predictions = model(features)

        # print("\nknockout")
        features[:, policy_index] = 0
        knockout_predictions = model(features)

        knockout_diff = (base_predictions - knockout_predictions).detach().cpu().numpy()
        knockout_diff = knockout_diff[~np.isnan(knockout_diff)]
        means[-1][0] = np.mean(knockout_diff)
        stds[-1][0] = np.std(knockout_diff)
        cis[-1][0] = stats.sem(knockout_diff) * stats.t.ppf((1 + confidence) / 2., len(knockout_diff)-1)

        policies_same_category_predictions = []
        for i, active_index_diff in enumerate(index_differences):
            if active_index_diff == 0:
                continue

            # print(f"\nchanged policy ({yticks[policy_index+active_index_diff]})")
            # new policy from same category
            features[:, policy_index+active_index_diff] = 1

            knockout_predictions = model(features)

            knockout_diff = (base_predictions - knockout_predictions).detach().cpu().numpy()
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
    


def permutation_importance(model, X_test, Y_test, path, dataset=""):
    K = 50
    n_policies = 46

    baseline_score = model.score(X_test, Y_test, mc_dropout_samples=1000)
    np.save(path + "permutation_importance_baseline_score.npy", np.array([baseline_score]))
    print(baseline_score)

    importances = np.zeros(shape=(n_policies, K))

    for policy_index in range(n_policies):
        print(f"policy_index {policy_index}")
        shuffled_scores = np.zeros(K)
        for k in range(K):
            print(f"k = {k}")
            X_shuffled = X_test.clone().detach()
            indices = np.arange(X_shuffled.shape[0])
            np.random.shuffle(indices)
            X_shuffled[:, policy_index] = X_shuffled[:, policy_index][indices]
            
            shuffled_scores[k] = model.score(X_shuffled, Y_test, mc_dropout_samples=1000)
            print(shuffled_scores[k])

        importances[policy_index] = baseline_score - shuffled_scores

    print(baseline_score)
    print(np.mean(importances, axis=1))
    print(np.std(importances, axis=1))

    np.save(path + "permutation_importances.npy", importances)
    importances = np.load(path + "permutation_importances.npy")
    
    if dataset == "":
        yticks = np.array(["C1 1","C1 2","C1 3","C2 1","C2 2","C2 3","C3 1","C3 2","C4 1","C4 2","C4 3","C4 4","C5 1","C5 2","C6 1","C6 2","C6 3","C7 1","C7 2","C8 1","C8 2","C8 3","C8 4","E1 1","E1 2","E2 1","E2 2","H1 1","H1 2","H2 1","H2 2","H2 3","H3 1","H3 2","H6 1","H6 2","H6 3","H6 4","H7 1","H7 2","H7 3","H7 4","H7 5","H8 1","H8 2","H8 3",])
    else:
        yticks = np.array(["ActivateCaseNotification","ActivateOrEstablishEmergencyResponse","ActivelyCommunicateWithHealthcareProfessionals1","ActivelyCommunicateWithManagers1","AdaptProceduresForPatientManagement","AirportHealthCheck","AirportRestriction","BorderHealthCheck","BorderRestriction","ClosureOfEducationalInstitutions","CordonSanitaire","CrisisManagementPlans","EducateAndActivelyCommunicateWithThePublic1","EnhanceDetectionSystem","EnhanceLaboratoryTestingCapacity","EnvironmentalCleaningAndDisinfection","IncreaseAvailabilityOfPpe","IncreaseHealthcareWorkforce","IncreaseInMedicalSuppliesAndEquipment","IncreaseIsolationAndQuarantineFacilities","IncreasePatientCapacity","IndividualMovementRestrictions","IsolationOfCases","MassGatheringCancellation","MeasuresForPublicTransport","MeasuresForSpecialPopulations","MeasuresToEnsureSecurityOfSupply","NationalLockdown","PersonalProtectiveMeasures","PoliceAndArmyInterventions","PortAndShipRestriction","ProvideInternationalHelp","PublicTransportRestriction","Quarantine","ReceiveInternationalHelp","RepurposeHospitals","Research","RestrictedTesting","ReturnOperationOfNationals","SmallGatheringCancellation","SpecialMeasuresForCertainEstablishments","Surveillance","TheGovernmentProvideAssistanceToVulnerablePopulations","TracingAndTracking","TravelAlertAndWarning","WorkSafetyProtocols",])
    
    sorted_indices = np.argsort(np.mean(importances, axis=1))

    fig, ax = plt.subplots(figsize=(12,12))
    ax.boxplot(importances[sorted_indices,:].T, vert=False, labels=yticks[sorted_indices])
    plt.savefig(path + "permutation_importance.png")
    plt.clf()
    plt.close()

    # plt.show()





def main():
    dataset = ""
    
    # dm = ResponseDataModule()
    # pt = PolicyTrackerProduct()

    # callbacks = [
    #     # save model with lowest validation loss
    #     pl.callbacks.ModelCheckpoint(monitor="val_loss"),
    #     # stop when validation loss stops decreasing
    #     pl.callbacks.EarlyStopping(monitor="val_loss", patience=10),
    # ]

    # logger = TensorBoardLogger(save_dir="lightning_logs", name="", default_hp_metric=False, log_graph=True)

    # trainer = pl.Trainer(
    #     max_epochs=200,
    #     callbacks=callbacks,
    #     logger=logger,
    #     gpus=1,
    # )

    # process = tensorboard()

    # trainer.fit(pt, datamodule=dm)

    # print(f"finished training, R0 = {pt.R0_model.weight.data}")
    
    # trainer.save_checkpoint("./ProductAnsatz/model_product_dropout.ckpt")

    # train_indices = np.array(dm.train_ds.indices)
    # val_indices = np.array(dm.val_ds.indices)
    # np.save("./ProductAnsatz/train_indices.npy", train_indices)
    # np.save("./ProductAnsatz/val_indices.npy", val_indices)


    checkpoint = "./ProductAnsatz/model_product_dropout.ckpt"#latest_checkpoint()
    pt = PolicyTrackerProduct.load_from_checkpoint(checkpoint)
    pt.to('cuda')
    pt.eval()

    print(pt.R0_model.weight)

    dm = ResponseDataModule()
    dm.prepare_data()
    dm.setup()
    train_indices = np.load("./ProductAnsatz/train_indices.npy")
    val_indices = np.load("./ProductAnsatz/val_indices.npy")
    train_indices = np.array(dm.train_ds.indices)
    val_indices = np.array(dm.val_ds.indices)
    val_features, val_responses = dm.val_ds.dataset.tensors
    val_features = val_features[val_indices].to('cuda')
    val_responses = val_responses[val_indices]

    knockout_evaluation(pt, "./ProductAnsatz/", vaccination_rate="any")
    # knockout_evaluation(pt, "./ProductAnsatz/", vaccination_rate="zero")
    # knockout_evaluation(pt, "./ProductAnsatz/", vaccination_rate="nonzero")
    permutation_importance(pt, val_features, val_responses, "./ProductAnsatz/")

    # # knockout_evaluation_same_category(pt, dataset=dataset)

    countries = ("Germany", "Spain", "Italy", "Japan", "Australia", "Argentina")

    # plot_countries(pt, countries, randomize_policies=True)
    plot_countries(pt, "./ProductAnsatz/", countries, randomize_policies=False)

    # plot_single_policy()
    # plot_policies_vaccination(0)
    # plot_policies_vaccination(1)


    # print("Press Enter to terminate Tensorboard.")
    # input()

    # process.terminate()



if __name__ == "__main__":
    main()
