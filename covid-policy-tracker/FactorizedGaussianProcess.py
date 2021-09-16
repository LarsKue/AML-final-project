import numpy as np
import matplotlib.pyplot as plt
from GPy.core.model import Model
import GPy
import pickle
import pandas as pd


import pathlib
import re

from datamodule import ResponseDataModule



class FactorizedGaussianProcess(Model):
    def __init__(self, X, Y, M, normalize_X=True, normalize_Y=True, partition_type="disjoint", kernel=None):
        super(FactorizedGaussianProcess, self).__init__("FactorizedGaussianProcess")

        self.X = X
        self.Y = Y
        self.M = M

        self.normalize_Y = normalize_Y
        self.normalize_X = normalize_X

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

        if self.normalize_X:
            self.X_mean = np.mean(self.X, axis=0)
            self.X_std = np.std(self.X, axis=0)
            self.X = (self.X - self.X_mean) / self.X_std

        if self.normalize_Y:
            self.Y_mean = np.mean(self.Y, axis=0)
            self.Y_std = np.std(self.Y, axis=0)
            self.Y = (self.Y - self.Y_mean) / self.Y_std


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
            
            for k in range(self.M):
                if k == communication_index:
                    continue
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
            

        else:
            raise ValueError(f"unknown aggregation type: {aggregation_type}")


        if self.normalize_Y:
            mean = mean * self.Y_std + self.Y_mean
            std = std * self.Y_std**2
            

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


def plot_countries(model, path, countries=("Germany",), randomize_policies=False, aggregation_type="rBCM"):
    df = pd.read_csv("policies_onehot_full.csv")

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
        if -.5 < new_ylims[0] < .5:
            ylims.append(new_ylims[0])
        if -.5 < new_ylims[1] < .5:
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
    version = latest_version(pathlib.Path("./FactorizedGaussianProcesses/"))
    if version is None:
        version = 0
    else:
        version += 1
    print(version)
    pathlib.Path(f"./FactorizedGaussianProcesses/version_{version}").mkdir()


    dm = ResponseDataModule()
    dm.prepare_data()
    dm.setup()

    train_features, train_responses = dm.train_ds.dataset.tensors
    train_features = train_features.detach().numpy()[dm.train_ds.indices]
    train_responses = train_responses.detach().numpy()[dm.train_ds.indices]
    
    val_features, val_responses = dm.val_ds.dataset.tensors
    val_features = val_features.detach().numpy()[dm.val_ds.indices]
    val_responses = val_responses.detach().numpy()[dm.val_ds.indices]


    train_indices = np.array(dm.train_ds.indices)
    val_indices = np.array(dm.val_ds.indices)
    np.save(f"./FactorizedGaussianProcesses/version_{version}/train_indices.npy", train_indices)
    np.save(f"./FactorizedGaussianProcesses/version_{version}/val_indices.npy", val_indices)

    print(train_features.shape, train_responses.shape)
    print(val_features.shape, val_responses.shape)

    model = FactorizedGaussianProcess(train_features, train_responses, 15, normalize_Y=False)
    print(model)

    model.optimize()
    print(model)
    

    with open(f"./FactorizedGaussianProcesses/version_{version}/factorizedGPr.dump" , "wb") as f:
        pickle.dump(model, f)
    

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
        

    countries = ("Germany", "Spain", "Italy", "Japan", "Australia", "Argentina")
    for aggregation_type in ["mean", "SPV", "PoE", "GPoE", "GPoE_constant_beta", "BCM", "rBCM", "grBCM"]:
        val_pred_mean, val_pred_var = model.predict(val_features, aggregation_type=aggregation_type)
        np.save(f"./FactorizedGaussianProcesses/version_{version}/val_set_prediction_{aggregation_type}.npy", np.hstack((val_pred_mean, val_pred_var, val_responses)))
        
        plot_countries(model, path=f"./FactorizedGaussianProcesses/version_{version}/", countries=countries, randomize_policies=False, aggregation_type=aggregation_type)
        
        plot_policies_vaccination(model, 0, path=f"./FactorizedGaussianProcesses/version_{version}/", aggregation_type=aggregation_type)
        
    for aggregation_type in ["NPAE"]:
        plot_policies_vaccination(model, 0, path=f"./FactorizedGaussianProcesses/version_{version}/", aggregation_type=aggregation_type)
        
        df = pd.read_csv("policies_onehot_full.csv")
        for i, country in enumerate(countries):
            plot_country(model, df, path=f"./FactorizedGaussianProcesses/version_{version}/", country=country, randomize_policies=False, aggregation_type=aggregation_type)

        val_pred_mean, val_pred_var = model.predict(val_features, aggregation_type=aggregation_type)
        np.save(f"./FactorizedGaussianProcesses/version_{version}/val_set_prediction_{aggregation_type}.npy", np.hstack((val_pred_mean, val_pred_var, val_responses)))
        

    #plt.show()


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
