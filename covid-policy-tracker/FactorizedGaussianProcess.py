import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from GPy.core.model import Model
import GPy
import pickle
import pandas as pd
import time

import pathlib
import re

from datamodule import ResponseDataModule



class FactorizedGaussianProcess(Model):
    def __init__(self, X, Y, M, normalize_X=False, normalize_Y=False, partition_type="disjoint", kernel=None):
        super(FactorizedGaussianProcess, self).__init__("FactorizedGaussianProcess")

        self.X = X
        self.Y = Y
        self.M = M

        self.normalize_X = normalize_X
        self.normalize_Y = normalize_Y

        self.N = X.shape[0]
        self.n = self.N // self.M

        # cached variables for NPAE prediction
        self.NPAE_K_cross_cache = None
        self.NPAE_K_invs_cache = None
        

        if partition_type == "random":
            self.partition = np.random.choice(self.N, size=(self.M, self.n), replace=True)
        elif partition_type == "disjoint":
            indices = np.arange(self.N)
            np.random.shuffle(indices)
            self.partition = np.array_split(indices, self.M)
        else:
            raise ValueError(f"Unknown partition type: {partition_type}")

        if kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=self.X.shape[1]) + GPy.kern.White(input_dim=self.X.shape[1])
        else:
            self.kernel = kernel


        if self.normalize_Y:
            self.Y_mean = np.mean(self.Y, axis=0)
            self.Y_std = np.std(self.Y, axis=0)
            self.Y = (self.Y - self.Y_mean) / self.Y_std

        if self.normalize_X:
            self.X_mean = np.mean(self.X, axis=0)
            self.X_std = np.std(self.X, axis=0)
            self.X = (self.X - self.X_mean) / self.X_std



        # same hyperparameters for all experts
        self.gpr = GPy.models.GPRegression(self.X[self.partition[0],:], self.Y[self.partition[0],:], kernel=self.kernel)
        self.link_parameters(self.gpr)



    def log_likelihood_k(self, k):
        X_k = self.X[self.partition[k],:]
        Y_k = self.Y[self.partition[k],:]

        self.gpr.set_XY(X_k, Y_k)

        return self.gpr.log_likelihood()

    def log_likelihood(self):
        print("log_likelihood")
        res = np.zeros(self.M)
        for k in range(self.M):
            res[k] = self.log_likelihood_k(k)
        
        return np.sum(res)

    def _log_likelihood_gradients_k(self, k):
        X_k = self.X[self.partition[k],:]
        Y_k = self.Y[self.partition[k],:]

        self.gpr.set_XY(X_k, Y_k)

        return self.gpr._log_likelihood_gradients()

    def _log_likelihood_gradients(self):
        print("log_likelihood_gradients")
        res = np.zeros(shape=(self.M, len(self.gradient)))
        for k in range(self.M):
            res[k] = self._log_likelihood_gradients_k(k)

        return np.sum(res, axis=0)

    def predict_k(self, k, X_test):
        X_k = self.X[self.partition[k],:]
        Y_k = self.Y[self.partition[k],:]

        self.gpr.set_XY(X_k, Y_k)

        prediction_k = self.gpr.predict(X_test)
        
        return prediction_k

    def predict_k_communication(self, k, communication_index, X_test):
        X_k = self.X[self.partition[k],:]
        Y_k = self.Y[self.partition[k],:]
        
        X_c = self.X[self.partition[communication_index],:]
        Y_c = self.Y[self.partition[communication_index],:]

        X_plus = np.vstack((X_k, X_c))
        Y_plus = np.vstack((Y_k, Y_c))


        self.gpr.set_XY(X_plus, Y_plus)
        
        prediction_k = self.gpr.predict(X_test)
        
        return prediction_k

    def NPAE_cross_kernel_and_inverses(self):
        kernel_function = self.gpr.kern.K
        gpr_variance = self.gpr.likelihood.variance

        K_cross = []
        K_invs = []
        for i in range(self.M):
            K_cross.append([])
            for j in range(self.M):
                print(f"cross kernel ({i},{j})")
                if i == j:
                    self_covariance = kernel_function(self.X[self.partition[i],:]) + gpr_variance
                    K_cross[-1].append(self_covariance)
                    K_invs.append(np.linalg.inv(self_covariance))
                else:
                    K_cross[-1].append(kernel_function(self.X[self.partition[i],:], self.X[self.partition[j],:]))
                    
        return K_invs, K_cross

    def NPAE_covariance_vector(self, x, K_invs):
        kernel_function = self.gpr.kern.K
        k_M_x = np.zeros(self.M)
        k_x_Xi_list = []

        for i in range(self.M):
            k_x_Xi = kernel_function(x[np.newaxis, :], self.X[self.partition[i],:])
            k_x_Xi_list.append(k_x_Xi)
            k_M_x[i] = k_x_Xi.dot(K_invs[i].dot(k_x_Xi.T))

        return k_x_Xi_list, k_M_x
    
    def NPAE_covariance_matrix_inverse(self, x, K_invs, K_cross, k_x_Xi_list):
        kernel_function = self.gpr.kern.K
        K_M_x = np.zeros(shape=(self.M, self.M))

        # calculate only upper triangular matrix
        for i in range(self.M):
            for j in range(i, self.M):
                print(f"K_M_x ({i},{j})")
                k_x_Xi = k_x_Xi_list[i]
                k_Xj_x = k_x_Xi_list[j].T
                K_Xi_Xj = K_cross[i][j]

                if i == j:
                    # 0.5 because of adding lower triangular matrix later
                    K_M_x[i, j] = 0.5 * k_x_Xi.dot(K_invs[i].dot(K_Xi_Xj.dot(K_invs[j].dot(k_Xj_x))))
                else:
                    K_M_x[i, j] = k_x_Xi.dot(K_invs[i].dot(K_Xi_Xj.dot(K_invs[j].dot(k_Xj_x))))

        # add lower triangular part
        K_M_x = K_M_x + K_M_x.T

        return np.linalg.inv(K_M_x + np.eye(K_M_x.shape[0])*1e-10)

    def predict(self, X_test, aggregation_type="rBCM"):
        if self.normalize_X:
            X_test = (X_test.copy() - self.X_mean) / self.X_std
        
        
        print(f"predicting - {aggregation_type}")
        mean = np.zeros(shape=(X_test.shape[0], 1))
        inverse_variance = np.zeros(shape=(X_test.shape[0], 1))
        variance = np.zeros(shape=(X_test.shape[0], 1))

        if aggregation_type == "mean":
            for k in range(self.M):
                print(f"expert {k+1}/{self.M}")
                prediction_k = self.predict_k(k, X_test)
                mean += prediction_k[0] / self.M
                variance += prediction_k[1] / self.M

        elif aggregation_type == "SPV":
            variance = np.ones(shape=(X_test.shape[0], 1)) * np.inf
            
            for k in range(self.M):
                print(f"expert {k+1}/{self.M}")
                mean_k, variance_k = self.predict_k(k, X_test)

                variance_stack = np.hstack((variance, variance_k))
                mean_stack = np.hstack((mean, mean_k))
                indices = np.argmin(variance_stack, axis=1)
                arange = np.arange(variance_stack.shape[0])

                mean = mean_stack[arange, indices][:, np.newaxis]
                variance = variance_stack[arange, indices][:, np.newaxis]

        elif aggregation_type == "PoE":
            for k in range(self.M):
                print(f"expert {k+1}/{self.M}")
                prediction_k = self.predict_k(k, X_test)
                mean += prediction_k[0] / prediction_k[1]
                inverse_variance += 1 / prediction_k[1]

            mean = mean / inverse_variance
            variance = 1 / inverse_variance

        elif aggregation_type == "GPoE":
            prior_variance = np.diag(self.gpr.kern.K(X_test)).reshape(-1,1) + self.gpr.likelihood.variance

            for k in range(self.M):
                print(f"expert {k+1}/{self.M}")
                prediction_k = self.predict_k(k, X_test)
                beta_k = 0.5 * (np.log(prior_variance) - np.log(prediction_k[1])).reshape(-1,1)
                # beta_k = 1 / self.M

                mean += beta_k * prediction_k[0] / prediction_k[1]
                inverse_variance += beta_k / prediction_k[1]

            mean = mean / inverse_variance
            variance = 1 / inverse_variance

        elif aggregation_type == "GPoE_constant_beta":
            prior_variance = np.diag(self.gpr.kern.K(X_test)).reshape(-1,1) + self.gpr.likelihood.variance

            for k in range(self.M):
                print(f"expert {k+1}/{self.M}")
                prediction_k = self.predict_k(k, X_test)
                beta_k = 1 / self.M

                mean += beta_k * prediction_k[0] / prediction_k[1]
                inverse_variance += beta_k / prediction_k[1]

            mean = mean / inverse_variance
            variance = 1 / inverse_variance

        elif aggregation_type == "BCM":
            prior_variance = np.diag(self.gpr.kern.K(X_test)).reshape(-1,1) + self.gpr.likelihood.variance
            inverse_variance += (1 - self.M) / prior_variance

            for k in range(self.M):
                print(f"expert {k+1}/{self.M}")
                prediction_k = self.predict_k(k, X_test)
                mean += prediction_k[0] / prediction_k[1]
                inverse_variance += 1 / prediction_k[1]
                
            mean = mean / inverse_variance
            variance = 1 / inverse_variance

        elif aggregation_type == "rBCM":
            prior_variance = np.diag(self.gpr.kern.K(X_test)).reshape(-1,1) + self.gpr.likelihood.variance
            beta_k_sum = np.zeros(shape=(X_test.shape[0], 1))
            for k in range(self.M):
                print(f"expert {k+1}/{self.M}")
                prediction_k = self.predict_k(k, X_test)
                beta_k = 0.5 * (np.log(prior_variance) - np.log(prediction_k[1])).reshape(-1,1)

                mean += beta_k * prediction_k[0] / prediction_k[1]
                inverse_variance += beta_k / prediction_k[1]
                beta_k_sum += beta_k

            inverse_variance += (1 - beta_k_sum) / prior_variance
            mean = mean / inverse_variance
            variance = 1 / inverse_variance

        elif aggregation_type == "grBCM":
            prior_variance = np.diag(self.gpr.kern.K(X_test)).reshape(-1,1) + self.gpr.likelihood.variance
            beta_k_sum = np.zeros(shape=(X_test.shape[0], 1))
            
            communication_index = 0
            prediction_c = self.predict_k(communication_index, X_test)
            
            for k in range(1, self.M):
                print(f"expert {k+1}/{self.M}")
                prediction_k_plus = self.predict_k_communication(k, communication_index, X_test)
                if k == 1:
                    beta_k = 1
                else:
                    beta_k = 0.5 * (np.log(prediction_c[1]) - np.log(prediction_k_plus[1])).reshape(-1,1)

                mean += beta_k * prediction_k_plus[0] / prediction_k_plus[1]
                inverse_variance += beta_k / prediction_k_plus[1]
                beta_k_sum += beta_k

            inverse_variance += (1 - beta_k_sum) / prediction_c[1]
            mean += (1 - beta_k_sum) * prediction_c[0] / prediction_c[1]
            mean = mean / inverse_variance
            variance = 1 / inverse_variance

        elif aggregation_type == "NPAE":
            prior_variance = np.diag(self.gpr.kern.K(X_test)).reshape(-1,1) + self.gpr.likelihood.variance
            
            K_invs, K_cross = self.NPAE_cross_kernel_and_inverses()
            
            all_means = np.zeros(shape=(len(X_test), self.M))
            for i in range(self.M):
                print(f"expert {i+1}/{self.M}")
                all_means[:, i] = self.predict_k(i, X_test)[0].squeeze()

            for i in range(len(X_test)):
                if i % 10 == 0:
                    print(f"test point {i}/{len(X_test)}")

                x = X_test[i]
                M_x = all_means[i]

                k_x_Xi_list, k_M_x = self.NPAE_covariance_vector(x, K_invs)
                K_M_x_inv = self.NPAE_covariance_matrix_inverse(x, K_invs, K_cross, k_x_Xi_list)

                mean[i] = k_M_x.T.dot(K_M_x_inv.dot(M_x))
                variance[i] = prior_variance[i] - k_M_x.T.dot(K_M_x_inv.dot(k_M_x))
            
        elif aggregation_type == "all":
            prior_variance = self.gpr.kern.Kdiag(X_test).reshape(-1,1) + self.gpr.likelihood.variance
            
            num_aggregation_types = 8
            
            means = np.zeros(shape=(num_aggregation_types, X_test.shape[0], 1))
            inverse_variances = np.zeros(shape=(num_aggregation_types, X_test.shape[0], 1))
            variances = np.zeros(shape=(num_aggregation_types, X_test.shape[0], 1))
            
            # SPV
            variances[1] = np.ones(shape=(X_test.shape[0], 1)) * np.inf

            # BCM
            inverse_variances[5] = (1 - self.M) / prior_variance

            # rBCM
            beta_k_sum_rBCM = np.zeros(shape=(X_test.shape[0], 1))

            # grBCM
            beta_k_sum_grBCM = np.zeros(shape=(X_test.shape[0], 1))
            communication_index = 0
            prediction_c = None
            

            for k in range(self.M):
                print(f"expert {k+1}/{self.M}")
                prediction_k = self.predict_k(k, X_test)

                # mean
                # print(f"\tmean")
                means[0] += prediction_k[0] / self.M
                variances[0] += prediction_k[1] / self.M

                # SPV
                # print(f"\tSPV")
                variance_stack = np.hstack((variances[1], prediction_k[1]))
                mean_stack = np.hstack((mean, prediction_k[0]))
                indices = np.argmin(variance_stack, axis=1)
                arange = np.arange(variance_stack.shape[0])
                means[1] = mean_stack[arange, indices][:, np.newaxis]
                variances[1] = variance_stack[arange, indices][:, np.newaxis]

                # PoE
                # print(f"\tPoE")
                means[2] += prediction_k[0] / prediction_k[1]
                inverse_variances[2] += 1 / prediction_k[1]
                
                # GPoE
                # print(f"\tGPoE")
                beta_k = 0.5 * (np.log(prior_variance) - np.log(prediction_k[1])).reshape(-1,1)
                means[3] += beta_k * prediction_k[0] / prediction_k[1]
                inverse_variances[3] += beta_k / prediction_k[1]
                
                # GPoE_constant_beta
                # print(f"\tGPoE_constant_beta")
                beta_k = 1 / self.M
                means[4] += beta_k * prediction_k[0] / prediction_k[1]
                inverse_variances[4] += beta_k / prediction_k[1]
                
                # BCM
                # print(f"\tBCM")
                means[5] += prediction_k[0] / prediction_k[1]
                inverse_variances[5] += 1 / prediction_k[1]
                
                # rBCM
                # print(f"\trBCM")
                beta_k = 0.5 * (np.log(prior_variance) - np.log(prediction_k[1])).reshape(-1,1)
                means[6] += beta_k * prediction_k[0] / prediction_k[1]
                inverse_variances[6] += beta_k / prediction_k[1]
                beta_k_sum_rBCM += beta_k
                
                # # grBCM
                # # print(f"\tgrBCM")
                # if k == 0:
                #     prediction_c = prediction_k
                # else:
                #     prediction_k_plus = self.predict_k_communication(k, communication_index, X_test)
                #     if k == 1:
                #         beta_k = 1
                #     else:
                #         beta_k = 0.5 * (np.log(prediction_c[1]) - np.log(prediction_k_plus[1])).reshape(-1,1)

                #     means[7] += beta_k * prediction_k_plus[0] / prediction_k_plus[1]
                #     inverse_variances[7] += beta_k / prediction_k_plus[1]
                #     beta_k_sum_grBCM += beta_k

            
            # PoE
            means[2] = means[2] / inverse_variances[2]
            variances[2] = 1 / inverse_variances[2]
            
            # GPoE
            means[3] = means[3] / inverse_variances[3]
            variances[3] = 1 / inverse_variances[3]
            
            # GPoE_constant_beta
            means[4] = means[4] / inverse_variances[4]
            variances[4] = 1 / inverse_variances[4]
            
            # BCM
            means[5] = means[5] / inverse_variances[5]
            variances[5] = 1 / inverse_variances[5]
            
            # rBCM
            inverse_variances[6] += (1 - beta_k_sum_rBCM) / prior_variance
            means[6] = means[6] / inverse_variances[6]
            variances[6] = 1 / inverse_variances[6]
            
            # # grBCM
            # inverse_variances[7] += (1 - beta_k_sum_grBCM) / prediction_c[1]
            # means[7] += (1 - beta_k_sum_grBCM) * prediction_c[0] / prediction_c[1]
            # means[7] = means[7] / inverse_variances[7]
            # variances[7] = 1 / inverse_variances[7]

            return means, variances

        else:
            raise ValueError(f"unknown aggregation type: {aggregation_type}")


        if self.normalize_Y:
            mean = mean * self.Y_std + self.Y_mean
            variance = variance * self.Y_std**2
            

        return mean, variance
        


def plot_country(model, df, path, country="Germany", randomize_policies=False, aggregation_type="rBCM"):
    df = df[df["country"] == country]

    df.pop("country")

    y = df.pop("reproduction_rate").to_numpy()[..., np.newaxis]
    x = df.to_numpy()

    # drop index
    x = x[:, 1:]

    if randomize_policies:
        # this is useful to check how much the model relies on this vs other features
        n_policies = 46
        random_x = np.random.randint(0, 1, size=(x.shape[0], n_policies))
        x[:, :n_policies] = random_x

    mu, sigma_sqr = model.predict(x, aggregation_type=aggregation_type)
    mu = mu.squeeze()
    sigma_sqr = sigma_sqr.squeeze()
    sigma = np.sqrt(sigma_sqr)

    if not randomize_policies:
        prediction = np.stack((mu, sigma)).T
        np.save(path + f"country_{country}_{aggregation_type}.npy", prediction)

    if aggregation_type == "NPAE":
        return
    
    ax = plt.gca()
    ax.plot(np.arange(len(y)), y, label="Actual")

    ax.plot(np.arange(len(y)), mu, label="Predicted")
    ax.fill_between(np.arange(len(y)), mu - 1.96*sigma, mu + 1.96*sigma, color="C0", alpha=0.2, label="95% confidence")

    ax.set_xlabel("Time")
    ax.set_ylabel("R")
    ax.set_title(country)
    ax.legend()


def plot_countries(model, path, countries=("Germany",), randomize_policies=False, aggregation_type="rBCM", dataset=""):
    df = pd.read_csv(dataset + "policies_onehot_full_absolute_R.csv")

    nrows = int(round(np.sqrt(len(countries))))
    ncols = len(countries) // nrows
    
    plt.figure(figsize=(6 * ncols + 1, 6 * nrows))

    axes = []
    for i, country in enumerate(countries):
        axes.append(plt.subplot(nrows, ncols, i + 1))
        plot_country(model, df, path=path, country=country, randomize_policies=randomize_policies, aggregation_type=aggregation_type)


    # set all ylims equal
    ylims = []
    for ax in axes:
        new_ylims = ax.get_ylim()
        # if -.5 < new_ylims[0] < .5:
        #     ylims.append(new_ylims[0])
        # if -.5 < new_ylims[1] < .5:
        #     ylims.append(new_ylims[1])
        ylims.append(new_ylims[0])
        ylims.append(new_ylims[1])

    ylims = [min(ylims), max(ylims)] if len(ylims) > 0 else [-.15, .11]
    for ax in axes:
        ax.set_ylim(ylims)

    if randomize_policies:
        plt.suptitle(aggregation_type + " (randomized policies)", fontsize=14)
        plt.savefig(path + f"fgp_countries_randomized_{aggregation_type}.png")
    else:
        plt.suptitle(aggregation_type, fontsize=14)
        plt.savefig(path + f"fgp_countries_{aggregation_type}.png")

    plt.close('all')


def plot_single_policy():
    checkpoint = latest_checkpoint()
    model = PolicyTracker.load_from_checkpoint(checkpoint)
    model.eval()

    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols)

    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)

        for j in range(6):
            policy = np.zeros(model.n_policies + model.n_other)
            policy[j] = 1

            x = np.tile(policy, (101, 1))
            x[:,-2] = 2*i * np.ones(len(x))
            x[:,-1] = np.linspace(0,1,101)

            x = torch.Tensor(x)

            y = model(x).detach().numpy()

            ax = plt.gca()
            ax.plot(np.linspace(0,1,101), y, label=j)
            #ax.set_xlabel("Vaccinations")
            #ax.set_ylabel("Delta R")
            ax.set_title(f"{2*i} days")
            #ax.legend()

    #plt.show()

def plot_policies_vaccination(model, vaccination, path, aggregation_type="rBCM"):
    n_policies = 46

    policies = np.eye(n_policies)

    x = np.zeros((n_policies+1, n_policies+2))
    x[1:,:-2] = policies
    x[:,-1] = vaccination * np.ones(n_policies+1)
    
    mu, sigma_sqr = model.predict(x, aggregation_type=aggregation_type)
    mu = mu.squeeze()
    sigma_sqr = sigma_sqr.squeeze()
    sigma = np.sqrt(sigma_sqr)

    prediction = np.stack((mu, sigma)).T
    np.save(path + f"single_policy_{aggregation_type}.npy", prediction)

    plt.figure(figsize=(19, 12))
    plt.errorbar(np.arange(n_policies+1), mu, yerr=sigma, fmt='.')

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
    plt.xticks(np.arange(n_policies+1), xticks, rotation='vertical')
    plt.title(aggregation_type)
    plt.savefig(path + f"fgp_single_policy_{aggregation_type}.png")
    
    #plt.show()


def knockout_evaluation(model, path, dataset=""):
    df = pd.read_csv(dataset + "policies_onehot_full_absolute_R.csv")
    df.pop("country")
    y = df.pop("reproduction_rate").to_numpy()[..., np.newaxis]
    x = df.to_numpy()
    # drop index
    x = x[:, 1:]

    n_policies = 46
    
    yticks = np.array([
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
    ])
    for policy_index in range(n_policies):
        mask = (x[:, policy_index] == 1)
        print(f"policy_index: {policy_index+1:2d}/{n_policies} -> {np.sum(mask):7d} instances\t({yticks[policy_index]})")


    means = np.zeros(shape=(8, n_policies))
    stds = np.zeros(shape=(8, n_policies))
    ci = np.zeros(shape=(8, n_policies))
    confidence = 0.95

    for policy_index in range(n_policies):
        mask = (x[:, policy_index] == 1)
        print(f"\npolicy_index: {policy_index+1}/{n_policies} -> {np.sum(mask):7d} instances\t({yticks[policy_index]})")
        
        if np.sum(mask) == 0:
            continue
        
        features = x[mask,:]
    
        print("\nbase")
        base_predictions = model.predict(features, aggregation_type="all")

        print("\nknockout")
        features[:, policy_index] = 0
        knockout_predictions = model.predict(features, aggregation_type="all")

        diff = base_predictions[0].squeeze() - knockout_predictions[0].squeeze()
        diff = diff[~np.isnan(diff).any(axis=1)]

        np.save(path + f"knockout/policy_{policy_index+1}.npy", diff)

        means[:, policy_index] = np.mean(diff, axis=1)
        stds[:, policy_index] = np.std(diff, axis=1)
        ci[:, policy_index] = stats.sem(diff, axis=1) * stats.t.ppf((1 + confidence) / 2., len(diff)-1)

        print(f"{yticks[policy_index]}\t{means[:, policy_index]}\t{stds[:, policy_index]}\t{ci[:, policy_index]}")


    np.save(path + f"knockout/means.npy", means)
    np.save(path + f"knockout/stds.npy", stds)
    np.save(path + f"knockout/CI.npy", ci)
    print(means.shape, means)
    print(stds.shape, stds)
    print(ci.shape, ci)


    for i, aggregation_type in enumerate(["mean", "SPV", "PoE", "GPoE", "GPoE_constant_beta", "BCM", "rBCM", "grBCM"]):
        plt.figure(figsize=(12, 12))
        plt.errorbar(means[i], -np.arange(n_policies), xerr=stds[i], fmt='.')
        plt.axvline(x=0.0, color="b")
        plt.yticks(-np.arange(n_policies), yticks, rotation='horizontal')
        for j, tick in enumerate(plt.gca().get_yticklabels()):
            tick.set_color("green" if means[i,j] < 0 else "red")
        plt.title(aggregation_type)
        plt.savefig(path + f"fgp_knockout_{aggregation_type}.png")
        plt.clf()
        plt.close()

        sorted_indices = np.argsort(means[i])
        plt.figure(figsize=(12, 12))
        plt.errorbar(means[i][sorted_indices], -np.arange(n_policies), xerr=stds[i][sorted_indices], fmt='.')
        plt.axvline(x=0.0, color="b")
        plt.yticks(-np.arange(n_policies), yticks[sorted_indices], rotation='horizontal')
        for j, tick in enumerate(plt.gca().get_yticklabels()):
            tick.set_color("green" if means[i][sorted_indices][j] < 0 else "red")
        plt.title(aggregation_type)
        plt.savefig(path + f"fgp_knockout_sorted_{aggregation_type}.png")
        plt.clf()
        plt.close()
        
        plt.figure(figsize=(12, 12))
        plt.errorbar(means[i], -np.arange(n_policies), xerr=ci[i], fmt='.')
        plt.axvline(x=0.0, color="b")
        plt.yticks(-np.arange(n_policies), yticks, rotation='horizontal')
        for j, tick in enumerate(plt.gca().get_yticklabels()):
            tick.set_color("green" if means[i,j] < 0 else "red")
        plt.title(aggregation_type)
        plt.savefig(path + f"fgp_knockout_ci_{aggregation_type}.png")
        plt.clf()
        plt.close()

        sorted_indices = np.argsort(means[i])
        plt.figure(figsize=(12, 12))
        plt.errorbar(means[i][sorted_indices], -np.arange(n_policies), xerr=ci[i][sorted_indices], fmt='.')
        plt.axvline(x=0.0, color="b")
        plt.yticks(-np.arange(n_policies), yticks[sorted_indices], rotation='horizontal')
        for j, tick in enumerate(plt.gca().get_yticklabels()):
            tick.set_color("green" if means[i][sorted_indices][j] < 0 else "red")
        plt.title(aggregation_type)
        plt.savefig(path + f"fgp_knockout_sorted_ci_{aggregation_type}.png")
        plt.clf()
        plt.close()


def knockout_evaluation_same_category(model, path, dataset=""):
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
        means.append(np.zeros(shape=(8, 1 + len(index_differences))))
        stds.append(np.zeros(shape=(8, 1 + len(index_differences))))
        cis.append(np.zeros(shape=(8, 1 + len(index_differences))))

        mask = (x[:, policy_index] == 1)
        # print(f"\npolicy_index: {policy_index+1}/{n_policies} -> {np.sum(mask):7d} instances\t({yticks[policy_index]})")
        
        if np.sum(mask) == 0:
            continue
            

        features = x[mask,:]
        print("\nbase")
        base_predictions = model.predict(features, aggregation_type="all")

        print("\nknockout")
        features[:, policy_index] = 0
        knockout_predictions = model.predict(features, aggregation_type="all")

        knockout_diff = base_predictions[0].squeeze() - knockout_predictions[0].squeeze()
        knockout_diff = knockout_diff[~np.isnan(knockout_diff).any(axis=1)]
        means[-1][:, 0] = np.mean(knockout_diff, axis=1)
        stds[-1][:, 0] = np.std(knockout_diff, axis=1)
        cis[-1][:, 0] = stats.sem(knockout_diff, axis=1) * stats.t.ppf((1 + confidence) / 2., len(knockout_diff)-1)

        for i, active_index_diff in enumerate(index_differences):
            if active_index_diff == 0:
                continue

            print(f"\nchanged policy ({yticks[policy_index]} -> {yticks[policy_index+active_index_diff]})")
            # new policy from same category
            features[:, policy_index+active_index_diff] = 1

            knockout_predictions = model.predict(features, aggregation_type="all")

            knockout_diff = base_predictions[0].squeeze() - knockout_predictions[0].squeeze()
            knockout_diff = knockout_diff[~np.isnan(knockout_diff).any(axis=1)]
            means[-1][:, i+1] = np.mean(knockout_diff, axis=1)
            stds[-1][:, i+1] = np.std(knockout_diff, axis=1)
            cis[-1][:, i+1] = stats.sem(knockout_diff, axis=1) * stats.t.ppf((1 + confidence) / 2., len(knockout_diff)-1)

            # reset
            features[:, policy_index+active_index_diff] = 0

        with np.printoptions(precision=4, suppress=True, linewidth=200):
            print(means[-1])


    
    for i, aggregation_type in enumerate(["mean", "SPV", "PoE", "GPoE", "GPoE_constant_beta", "BCM", "rBCM", "grBCM"]):
        plt.figure(figsize=(19,19))

        for policy_index, index_differences in enumerate(same_category_index_difference):
            ax = plt.subplot(6, 8, policy_index+1)
            
            ax.errorbar(means[policy_index][i], -np.arange(len(means[policy_index][i])), xerr=cis[policy_index][i], fmt=".")
            ax.axvline(x=0.0, color="black")

            ax.set_yticks(-np.arange(len(means[policy_index][i])))
            ax.set_yticklabels(["no"] + list(yticks[policy_index+index_differences[0]:policy_index+index_differences[-1]+1]))

            ax.set_title(yticks[policy_index])

        plt.suptitle(aggregation_type, fontsize=14)
        plt.subplots_adjust(hspace=1, wspace=.5)
        
        plt.savefig(path + f"fgp_knockout_same_category_{aggregation_type}.png")
        plt.clf()
        plt.close()
    

def latest_version(path):
    """
    Returns latest model version as integer
    """
    # unsorted
    versions = list(str(v.stem) for v in path.glob("version_*"))
    # get version numbers as integer
    versions = [re.match(r"version_(\d+)", version) for version in versions]
    versions = [int(match.group(1)) for match in versions]

    if len(versions) == 0:
        return None
    return max(versions)



def main():
    dataset = ""
    version = latest_version(pathlib.Path("./FactorizedGaussianProcesses/"))
    # if version is None:
    #     version = 0
    # else:
    #     version += 1
    # print(version)
    # pathlib.Path(f"./FactorizedGaussianProcesses/version_{version}").mkdir()


    # dm = ResponseDataModule()
    # dm.prepare_data()
    # dm.setup()

    # train_features, train_responses = dm.train_ds.dataset.tensors
    # train_features = train_features.detach().numpy()[dm.train_ds.indices]
    # train_responses = train_responses.detach().numpy()[dm.train_ds.indices]
    
    # val_features, val_responses = dm.val_ds.dataset.tensors
    # val_features = val_features.detach().numpy()[dm.val_ds.indices]
    # val_responses = val_responses.detach().numpy()[dm.val_ds.indices]


    # train_indices = np.array(dm.train_ds.indices)
    # val_indices = np.array(dm.val_ds.indices)
    # np.save(f"./FactorizedGaussianProcesses/version_{version}/train_indices.npy", train_indices)
    # np.save(f"./FactorizedGaussianProcesses/version_{version}/val_indices.npy", val_indices)

    # print(train_features.shape, train_responses.shape)
    # print(val_features.shape, val_responses.shape)

    # # kernel = None
    # kernel = GPy.kern.Matern32(input_dim=train_features.shape[1]) + GPy.kern.White(input_dim=train_features.shape[1])

    # model = FactorizedGaussianProcess(train_features, train_responses, 15, normalize_X=False, normalize_Y=True, kernel=kernel)
    # print(model)

    # model.optimize()
    # print(model)
    

    # with open(f"./FactorizedGaussianProcesses/version_{version}/factorizedGPr.dump" , "wb") as f:
    #     pickle.dump(model, f)
    

    print(version)
    train_indices = np.load(f"./FactorizedGaussianProcesses/version_{version}/train_indices.npy")
    val_indices = np.load(f"./FactorizedGaussianProcesses/version_{version}/val_indices.npy")

    dm = ResponseDataModule()
    dm.prepare_data()
    dm.setup()
    train_features, train_responses = dm.train_ds.dataset.tensors
    train_features = train_features.detach().numpy()[train_indices]
    train_responses = train_responses.detach().numpy()[train_indices]
    val_features, val_responses = dm.val_ds.dataset.tensors
    val_features = val_features.detach().numpy()[val_indices]
    val_responses = val_responses.detach().numpy()[val_indices]

    with open(f"./FactorizedGaussianProcesses/version_{version}/factorizedGPr.dump", "rb") as f:
        model = pickle.load(f)
        

    # countries = ("Germany", "Spain", "Italy", "Japan", "Australia", "Argentina")
    # for aggregation_type in ["mean", "SPV", "PoE", "GPoE", "GPoE_constant_beta", "BCM", "rBCM", "grBCM"]:
    #     val_pred_mean, val_pred_var = model.predict(val_features, aggregation_type=aggregation_type)
    #     np.save(f"./FactorizedGaussianProcesses/version_{version}/val_set_prediction_{aggregation_type}.npy", np.hstack((val_pred_mean, val_pred_var, val_responses)))
        
    #     #plot_countries(model, path=f"./FactorizedGaussianProcesses/version_{version}/", countries=countries, randomize_policies=True, aggregation_type=aggregation_type)

    #     plot_countries(model, path=f"./FactorizedGaussianProcesses/version_{version}/", countries=countries, randomize_policies=False, aggregation_type=aggregation_type)
        
    #     plot_policies_vaccination(model, 0, path=f"./FactorizedGaussianProcesses/version_{version}/", aggregation_type=aggregation_type)

    #knockout_evaluation(model, path=f"./FactorizedGaussianProcesses/version_{version}/", dataset=dataset)
    knockout_evaluation_same_category(model, path=f"./FactorizedGaussianProcesses/version_{version}/", dataset=dataset)

    # for aggregation_type in ["NPAE"]:
    #     plot_policies_vaccination(model, 0, path=f"./FactorizedGaussianProcesses/version_{version}/", aggregation_type=aggregation_type)

    #     # don't call plot_countries because of memory        
    #     df = pd.read_csv("policies_onehot_full.csv")
    #     for i, country in enumerate(countries):
    #         plot_country(model, df, path=f"./FactorizedGaussianProcesses/version_{version}/", country=country, randomize_policies=False, aggregation_type=aggregation_type)

    #     val_pred_mean, val_pred_var = model.predict(val_features, aggregation_type=aggregation_type)
    #     np.save(f"./FactorizedGaussianProcesses/version_{version}/val_set_prediction_{aggregation_type}.npy", np.hstack((val_pred_mean, val_pred_var, val_responses)))
        

    #plt.show()


def test():
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.sin(x) + np.random.randn(1000)*0.05
    x = x[:,None]
    y = y[:,None]

    train_idx = np.random.choice(len(x),size=800)

    model = FactorizedGaussianProcess(x[train_idx], y[train_idx], 20, normalize_X=False, normalize_Y=False)
    model.optimize()

    full = GPy.models.GPRegression(x[train_idx], y[train_idx])
    full.optimize()


    plt.figure(figsize=(19,12))
    for i, aggregation_type in enumerate(["SPV", "PoE", "GPoE", "GPoE_constant_beta", "BCM", "rBCM"]):
        mean, var = model.predict(x, aggregation_type=aggregation_type)
        mean_full, var_full = full.predict(x)

        std = np.sqrt(var)
        std_full = np.sqrt(var_full)

        l, u = mean - 1.96*std, mean + 1.96*std
        l_full, u_full = mean_full - 1.96*std_full, mean_full + 1.96*std_full

        plt.subplot(2,3, i+1)
        plt.plot(x, y, color="C0", label="Actual")
        plt.plot(x, mean, color="C1", label="Predicted")
        plt.plot(x, mean_full, color="C2", label="full")
        plt.fill_between(x.flat, l.flat, u.flat, color="C1", label="95%", alpha=0.2)
        plt.fill_between(x.flat, l_full.flat, u_full.flat, color="C2", label="95% full", alpha=0.2)
        plt.plot(x[train_idx], y[train_idx], linewidth=0, marker="x", color="C3", label="Train Points")
        plt.title(aggregation_type)
        plt.legend()

    plt.show()



if __name__ == "__main__":
    # version = 5
    # mses = np.zeros(7)
    # chi_squares = np.zeros(7)
    # for i, aggregation_type in enumerate(["mean", "SPV", "PoE", "GPoE", "GPoE_constant_beta", "BCM", "rBCM"]):
    #     val_data = np.load(f"./FactorizedGaussianProcesses/version_{version}/val_set_prediction_{aggregation_type}.npy")
    #     val_pred_mean = val_data[:,0]
    #     val_pred_var = val_data[:,1]
    #     val_responses = val_data[:,2]

    #     mse = ((val_pred_mean - val_responses)**2).mean()

    #     chi_square = ((val_pred_mean - val_responses)**2 / val_pred_var).sum()

    #     print(aggregation_type)
    #     print(mse)
    #     print(chi_square)

    #     mses[i] = mse
    #     chi_squares[i] = chi_square
        
    # print(["mean", "SPV", "PoE", "GPoE", "GPoE_constant_beta", "BCM", "rBCM"][np.argmin(mses)])
    # print(["mean", "SPV", "PoE", "GPoE", "GPoE_constant_beta", "BCM", "rBCM"][np.argmin(chi_squares)])
        

    main()
    # test()
