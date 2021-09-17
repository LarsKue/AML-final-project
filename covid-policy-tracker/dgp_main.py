
import numpy as np
import matplotlib.pyplot as plt
import dgp

from datamodule import ResponseDataModule


def main():

    dm = ResponseDataModule()
    dm.prepare_data()
    dm.setup()

    train_x, train_y = dm.train_ds.dataset.tensors
    val_x, val_y = dm.val_ds.dataset.tensors

    train_x = train_x.numpy()
    train_y = train_y.numpy()

    val_x = val_x.numpy()
    val_y = val_y.numpy()

    model = dgp.variants.ProductOfExperts(train_x, train_y, m=1000)

    model.optimize(max_iters=1)

    mean, var = model.predict(val_x)

    # TODO: plotting (below code is not designed to plot covid data)
    #  train with more than 1 iteration

    std = np.sqrt(var)

    l_95 = mean - 1.96 * std
    u_95 = mean + 1.96 * std

    plt.figure(figsize=(12, 9))
    plt.plot(train_x, train_y, label="Train", color="C0")
    plt.plot(val_x, mean, label="Validation", color="C1")
    plt.fill_between(val_x.flat, l_95.flat, u_95.flat, alpha=0.2, label="95% confidence", color="C1")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
