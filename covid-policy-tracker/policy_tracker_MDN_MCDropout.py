import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import ResponseDataModule

import subprocess as sub

import pathlib
import re

import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import datetime as dt


class PolicyTrackerMDN(pl.LightningModule):
    def __init__(self):
        super(PolicyTrackerMDN, self).__init__()

        self.num_gaussians = 3

        self.n_policies = 46
        self.n_other = 2

        self.example_input_array = torch.zeros(1, self.n_policies + self.n_other)

        self.base = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.PReLU(),
        )

        self.means = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, self.num_gaussians)
        )
        self.variances = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, self.num_gaussians)
        )
        self.weights = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, self.num_gaussians)
        )

        # perform forward pass to initialize lazy modules
        with torch.no_grad():
            self(self.example_input_array)

        # initialize weights
        for module in self.base.modules():
            if type(module) is nn.Linear:
                data = module.weight.data
                nn.init.normal_(data, 0, 2 / data.numel())

        
        def loss_fn(target, mu, sigma_sqr, pi):
            # pi, sigma, mu: (batch_size, num_gaussians)
            # print()
            # print("target", target.shape)
            # print("pi", pi.shape)
            # print("sigma_sqr", sigma_sqr.shape)
            # print("mu", mu.shape)

            exponents = -(target.expand_as(mu) - mu)**2 / (2*sigma_sqr)
            max_exponent = torch.max(exponents, dim=1).values
            # print(exponents)
            # print(max_exponent)
            # print(exponents.shape)
            # print(max_exponent.shape)
            # print(exponents - max_exponent.unsqueeze(1).expand_as(exponents))

            gaussian_prob = torch.exp(exponents - max_exponent.unsqueeze(1).expand_as(exponents)) / torch.sqrt(2*math.pi*sigma_sqr)

            # print("gaussian_prob", gaussian_prob.shape)
            # print("\n")
            # print(target)
            # print(mu)
            # print(sigma_sqr)
            # print(pi)
            # print(gaussian_prob)
            

            prob = pi * gaussian_prob
            prob[torch.isinf(gaussian_prob) & (pi < 1e-10)] = 0.0
            # print("prob", prob.shape)
            negative_log_likelihood = -torch.log(torch.sum(prob, dim=1)) - max_exponent
            # print("negative_log_likelihood", negative_log_likelihood.shape)
            
            return torch.mean(negative_log_likelihood)

        self.loss = loss_fn

    def forward(self, x, mc_dropout_samples=None, max_component_only=False):
        if mc_dropout_samples is None:
            z = self.base(x)

            mu = self.means(z)
            sigma = F.elu(self.variances(z)) + 1
            pi = F.softmax(self.weights(z), dim=1)

            return mu, sigma, pi
        else:
            is_training = self.training

            self.train()
            if max_component_only:
                means = torch.zeros(size=(x.shape[0], mc_dropout_samples))
                variances = torch.zeros(size=(x.shape[0], mc_dropout_samples))
                
                for i in range(mc_dropout_samples):
                    # if i % 100 == 0:
                    #     print(f"sample nr {i}/{mc_dropout_samples}")

                    mu, sigma_sqr, pi = self(x)

                    max_component = torch.argmax(pi, dim=1)
                    arange = torch.arange(len(max_component))

                    mu = mu[arange, max_component]
                    sigma_sqr = sigma_sqr[arange, max_component]

                    means[:, i] = mu.detach()
                    variances[:, i] = sigma_sqr.detach()

                mean_prediction = torch.mean(means, axis=1)
                variance_prediction = torch.mean(variances + torch.square(means), axis=1) - torch.square(mean_prediction)

                self.training = is_training
                return mean_prediction, variance_prediction
            else:
                means = torch.zeros(size=(mc_dropout_samples, x.shape[0], self.num_gaussians))
                variances = torch.zeros(size=(mc_dropout_samples, x.shape[0], self.num_gaussians))
                weights = torch.zeros(size=(mc_dropout_samples, x.shape[0], self.num_gaussians))

                for i in range(mc_dropout_samples):
                    # if i % 100 == 0:
                    #     print(f"sample nr {i}/{mc_dropout_samples}")

                    mu, sigma_sqr, pi = self(x)

                    means[i, :, :] = mu.detach()
                    variances[i, :, :] = sigma_sqr.detach()
                    weights[i, :, :] = pi.detach()

                self.training = is_training
                return means, variances, weights

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.3, threshold=0.05,
                                                                  verbose=True)
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor="val_loss")

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma_sqr, pi = self(x)

        loss = self.loss(y, mu, sigma_sqr, pi)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma_sqr, pi = self(x)

        loss = self.loss(y, mu, sigma_sqr, pi)

        self.log("val_loss", loss)

        return loss

    def score(self, X_test, Y_test, mc_dropout_samples=1000):
        batch_size = 100000
        num_batches = (len(X_test)+batch_size-1) // batch_size
        numerator = 0
        for i in range(num_batches):
            batch = X_test[i*batch_size:(i+1)*batch_size,:]
            means, _ = self(batch, mc_dropout_samples=mc_dropout_samples, max_component_only=True)
            numerator += ((Y_test[i*batch_size:(i+1)*batch_size,:].squeeze() - means)**2).sum()

        return 1 - (numerator / ((Y_test.squeeze() - Y_test.mean())**2).sum())




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

    means, variances = model(x, mc_dropout_samples=1000, max_component_only=True)
    
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


def plot_countries(model, path, countries=("Germany",), randomize_policies=False):
    df = pd.read_csv("policies_onehot_full_absolute_R.csv")

    nrows = int(round(np.sqrt(len(countries))))
    ncols = len(countries) // nrows
    
    plt.figure(figsize=(6 * ncols + 1, 6 * nrows))

    axes = []
    for i, country in enumerate(countries):
        print(country)
        axes.append(plt.subplot(nrows, ncols, i + 1))
        plot_country(model, df, country, randomize_policies=randomize_policies)


    # # set all ylims equal
    # ylims = []
    # for ax in axes:
    #     ylims.extend(ax.get_ylim())

    ylims = [0, 3.75]
    for ax in axes:
        ax.set_ylim(ylims)

    plt.savefig(path + "countries.png")
    plt.clf()
    plt.close()

    return ylims


def plot_country_heatmap(model, df, ylims, country="Germany", randomize_policies=False):
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

    means, variances, weights = model(x, mc_dropout_samples=1, max_component_only=False)
    print(means)
    print(variances)
    print(weights)

    num_ys = 1000
    
    y_grid = torch.linspace(ylims[0], ylims[1], num_ys)
    y_grid = y_grid.repeat(means.shape[1], 1)
    y_grid = y_grid.unsqueeze(2).repeat(1, 1, means.shape[2])

    total_prob = 0
    sample_size = means.shape[0]
    for i in range(sample_size):
        mu = means[i].unsqueeze(1).repeat(1, num_ys, 1)
        sigma_sqr = variances[i].unsqueeze(1).repeat(1, num_ys, 1)
        pi = weights[i].unsqueeze(1).repeat(1, num_ys, 1)

        exponents = -(y_grid - mu)**2 / (2*sigma_sqr)
        gaussian_prob = torch.exp(exponents) / torch.sqrt(2*math.pi*sigma_sqr)

        prob = pi * gaussian_prob
        prob = torch.sum(prob, dim=2)

        total_prob += prob / sample_size

    total_prob /= torch.max(total_prob)
    total_prob = total_prob.detach().cpu().numpy()


    ax = plt.gca()

    ax.plot(np.arange(len(y)), y.detach().cpu().numpy(), label="Actual", color="white")

    ax.imshow(total_prob.T, extent=[0, len(y), *ylims], origin="lower", aspect='auto', cmap="hot", interpolation='nearest')
    ax.autoscale(False)

    ax.set_xticks(first_day_indices)
    ax.set_xticklabels(dates_list, rotation=45)
    
    ax.set_ylabel("R")
    ax.set_title(country)
    ax.legend()

def plot_countries_heatmap(model, ylims, path, countries=("Germany",), randomize_policies=False):
    df = pd.read_csv("policies_onehot_full_absolute_R.csv")

    nrows = int(round(np.sqrt(len(countries))))
    ncols = len(countries) // nrows
    
    plt.figure(figsize=(6 * ncols + 1, 6 * nrows))

    axes = []
    for i, country in enumerate(countries):
        print(country)
        plt.subplot(nrows, ncols, i + 1)
        plot_country_heatmap(model, df, ylims, country, randomize_policies=randomize_policies)

    plt.savefig(path + "countries_heatmap.png")
    plt.show()
    plt.clf()
    plt.close()

    return ylims


def plot_single_policy(model):
    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols)

    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)

        for j in range(6):
            policy = np.zeros(model.n_policies + model.n_other)
            policy[j] = 1

            x = np.tile(policy, (101, 1))
            x[:, -2] = 2 * i * np.ones(len(x))
            x[:, -1] = np.linspace(0, 1, 101)

            x = torch.Tensor(x)

            mu, sigma_sqr, pi = model(x)
            max_component = torch.argmax(pi, dim=1)
            arange = torch.arange(len(max_component))
            mu = mu[arange, max_component].detach().cpu().numpy()
            sigma = torch.sqrt(sigma_sqr[arange, max_component]).detach().cpu().numpy()

            
            ax = plt.gca()
            ax.plot(np.linspace(0, 1, 101), mu, label=j)
            ax.fill_between(np.linspace(0, 1, 101), mu - sigma, mu + sigma, alpha=0.2)
            # ax.set_xlabel("Vaccinations")
            # ax.set_ylabel("Delta R")
            ax.set_title(f"{2 * i} days")
            # ax.legend()

    # plt.show()


def plot_policies_vaccination(model, vaccination):
    policies = np.eye(model.n_policies)

    x = np.zeros((model.n_policies + 1, model.n_policies + 2))
    x[1:, :-2] = policies
    x[:, -1] = vaccination * np.ones(model.n_policies + 1)
    x = torch.Tensor(x)

    means, variances = model.MC_Dropout(x, 1000)
    
    means = means.detach().cpu().numpy()
    stds = torch.sqrt(variances).detach().cpu().numpy()


    plt.figure()
    plt.errorbar(np.arange(model.n_policies + 1), means, yerr=stds, fmt='.')

    xticks = [
        "no",

        "C1 1",
        "C1 2",
        "C1 3",

        "C2 1",
        "C2 2",
        "C2 3",

        "C3 1",
        "C3 2",

        "C4 1",
        "C4 2",
        "C4 3",
        "C4 4",

        "C5 1",
        "C5 2",

        "C6 1",
        "C6 2",
        "C6 3",

        "C7 1",
        "C7 2",

        "C8 1",
        "C8 2",
        "C8 3",
        "C8 4",

        "E1 1",
        "E1 2",

        "E2 1",
        "E2 2",

        "H1 1",
        "H1 2",

        "H2 1",
        "H2 2",
        "H2 3",

        "H3 1",
        "H3 2",

        "H6 1",
        "H6 2",
        "H6 3",
        "H6 4",

        "H7 1",
        "H7 2",
        "H7 3",
        "H7 4",
        "H7 5",

        "H8 1",
        "H8 2",
        "H8 3",
    ]
    plt.xticks(np.arange(model.n_policies+1), xticks, rotation='vertical')
    
    # plt.show()


def knockout_evaluation(model, path, vaccination_rate="any", dataset=""):
    df = pd.read_csv(dataset + "policies_onehot_full_absolute_R.csv")
    df.pop("country")
    df.pop("dates")
    
    y = df.pop("reproduction_rate").to_numpy()[..., np.newaxis]
    x = df.to_numpy()
    # drop index
    x = x[:, 1:]
    x = torch.Tensor(x)

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


    if dataset == "":
        yticks = np.array(["C1 1","C1 2","C1 3","C2 1","C2 2","C2 3","C3 1","C3 2","C4 1","C4 2","C4 3","C4 4","C5 1","C5 2","C6 1","C6 2","C6 3","C7 1","C7 2","C8 1","C8 2","C8 3","C8 4","E1 1","E1 2","E2 1","E2 2","H1 1","H1 2","H2 1","H2 2","H2 3","H3 1","H3 2","H6 1","H6 2","H6 3","H6 4","H7 1","H7 2","H7 3","H7 4","H7 5","H8 1","H8 2","H8 3",])
    else:
        yticks = np.array(["ActivateCaseNotification","ActivateOrEstablishEmergencyResponse","ActivelyCommunicateWithHealthcareProfessionals1","ActivelyCommunicateWithManagers1","AdaptProceduresForPatientManagement","AirportHealthCheck","AirportRestriction","BorderHealthCheck","BorderRestriction","ClosureOfEducationalInstitutions","CordonSanitaire","CrisisManagementPlans","EducateAndActivelyCommunicateWithThePublic1","EnhanceDetectionSystem","EnhanceLaboratoryTestingCapacity","EnvironmentalCleaningAndDisinfection","IncreaseAvailabilityOfPpe","IncreaseHealthcareWorkforce","IncreaseInMedicalSuppliesAndEquipment","IncreaseIsolationAndQuarantineFacilities","IncreasePatientCapacity","IndividualMovementRestrictions","IsolationOfCases","MassGatheringCancellation","MeasuresForPublicTransport","MeasuresForSpecialPopulations","MeasuresToEnsureSecurityOfSupply","NationalLockdown","PersonalProtectiveMeasures","PoliceAndArmyInterventions","PortAndShipRestriction","ProvideInternationalHelp","PublicTransportRestriction","Quarantine","ReceiveInternationalHelp","RepurposeHospitals","Research","RestrictedTesting","ReturnOperationOfNationals","SmallGatheringCancellation","SpecialMeasuresForCertainEstablishments","Surveillance","TheGovernmentProvideAssistanceToVulnerablePopulations","TracingAndTracking","TravelAlertAndWarning","WorkSafetyProtocols",])
    

    n_policies = 46

    means = np.zeros(n_policies)
    # stds = np.zeros(n_policies)
    ci = np.zeros(n_policies)
    confidence = 0.95

    for policy_index in range(n_policies):
        print(f"policy_index: {policy_index+1}/{n_policies}")
        mask = (x[:, policy_index] == 1)
        
        if torch.sum(mask) == 0:
            continue
            
        diff = np.array([])
        
        features = x[mask,:].to('cuda')

        base_means, base_variances = model(features, mc_dropout_samples=1000, max_component_only=True)
        features[:, policy_index] = 0
        knockout_means, knockout_variances = model(features, mc_dropout_samples=1000, max_component_only=True)

        
        diff_means = (base_means - knockout_means).detach().cpu().numpy()
        diff_variances = (base_variances + knockout_variances).detach().cpu().numpy()

        print(diff_means.shape)
        mask = ~(np.isnan(diff_means) | np.isnan(diff_variances))
        diff_means = diff_means[mask]
        diff_variances = diff_variances[mask]

        means[policy_index] = np.mean(diff_means)

        variance = np.mean(diff_means**2 + diff_variances) - np.mean(diff_means)**2
        sem = np.sqrt(variance) / np.sqrt(len(diff_means))
        ci[policy_index] = sem * stats.t.ppf((1 + confidence) / 2., len(diff_means)-1)

        print(f"{yticks[policy_index]}\t{means[policy_index]}\t{ci[policy_index]}")


        # batch_size = 100000
        # num_batches = (len(features)+batch_size-1) // batch_size
        # for i in range(num_batches):
        #     batch = features[i*batch_size:(i+1)*batch_size,:]
            
        #     base_means, _ = model(batch, mc_dropout_samples=1000, max_component_only=True)

        #     batch[:, policy_index] = 0
        #     knockout_means, _ = model(batch, mc_dropout_samples=1000, max_component_only=True)


        #     diff_tmp = (base_means - knockout_means).detach().cpu().numpy()
        #     diff_tmp = diff_tmp[~np.isnan(diff_tmp)]

        #     diff = np.append(diff, diff_tmp)

        # means[policy_index] = np.mean(diff)
        # stds[policy_index] = np.std(diff)
        # ci[policy_index] = stats.sem(diff) * stats.t.ppf((1 + 0.95) / 2., len(diff)-1)


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
    plt.title("Mixture Density Network")
    plt.savefig(path + f"knockout_vaccination_{vaccination_rate}.png")
    plt.clf()
    plt.close()
        

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

    # importances = np.load(path + "permutation_importances.npy")

    
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






def main():
    dm = ResponseDataModule()
    
    pt = PolicyTrackerMDN()

    callbacks = [
        # save model with lowest validation loss
        pl.callbacks.ModelCheckpoint(monitor="val_loss"),
        # stop when validation loss stops decreasing
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=10),
    ]

    logger = TensorBoardLogger(save_dir="lightning_logs", name="", default_hp_metric=False, log_graph=True)

    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=callbacks,
        logger=logger,
        gpus=1,
    )

    process = tensorboard()

    trainer.fit(pt, datamodule=dm)
    trainer.save_checkpoint("./PolicyTrackerMDN/MC_Dropout/model_MDN_dropout.ckpt")

    train_indices = np.array(dm.train_ds.indices)
    val_indices = np.array(dm.val_ds.indices)
    np.save("./PolicyTrackerMDN/MC_Dropout/train_indices.npy", train_indices)
    np.save("./PolicyTrackerMDN/MC_Dropout/val_indices.npy", val_indices)


    checkpoint = "./PolicyTrackerMDN/MC_Dropout/model_MDN_dropout.ckpt"#latest_checkpoint()
    pt = PolicyTrackerMDN.load_from_checkpoint(checkpoint)
    pt.to('cuda')

    dm = ResponseDataModule()
    dm.prepare_data()
    dm.setup()
    train_indices = np.load("./PolicyTrackerMDN/MC_Dropout/train_indices.npy")
    val_indices = np.load("./PolicyTrackerMDN/MC_Dropout/val_indices.npy")
    train_indices = np.array(dm.train_ds.indices)
    val_indices = np.array(dm.val_ds.indices)
    val_features, val_responses = dm.val_ds.dataset.tensors
    val_features = val_features[val_indices].to('cuda')
    val_responses = val_responses[val_indices]

    pt.eval()

    knockout_evaluation(pt, "./PolicyTrackerMDN/MC_Dropout/", vaccination_rate="any")
    # knockout_evaluation(pt, "./PolicyTrackerMDN/MC_Dropout/", vaccination_rate="zero")
    # knockout_evaluation(pt, "./PolicyTrackerMDN/MC_Dropout/", vaccination_rate="nonzero")
    permutation_importance(pt, val_features, val_responses, "./PolicyTrackerMDN/MC_Dropout/")
    

    countries = ("Germany", "Spain", "Italy", "Japan", "Australia", "Argentina")

    # ylims = plot_countries(pt, countries=countries, randomize_policies=True)
    ylims = plot_countries(pt, "./PolicyTrackerMDN/MC_Dropout/", countries=countries, randomize_policies=False)

    # # plot_countries_heatmap(pt, ylims=ylims, countries=countries, randomize_policies=True)
    plot_countries_heatmap(pt, [0,3.75], "./PolicyTrackerMDN/MC_Dropout/", countries=countries, randomize_policies=False)

    # plot_policies_vaccination(model=pt, vaccination=0)
    # plot_policies_vaccination(model=pt, vaccination=1)


    # print("Press Enter to terminate Tensorboard.")
    # input()

    # process.terminate()


if __name__ == "__main__":
    main()
