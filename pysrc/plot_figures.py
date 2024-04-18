import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    cut = 2500
    imps = np.load("a2c_fetch/4/folder_10/avg_imp.npy")[:cut]
    sem = np.load("a2c_fetch/4/folder_10/sem_imp.npy")[:cut]
    plt.figure()
    upper_bounds = imps + sem
    lower_bounds = imps - sem
    iterations = np.arange(cut)

    # plt.errorbar(iterations, imps, yerr=sem, fmt="o", capsize=5, color='red')
    plt.plot(iterations, imps, color="orange")
    plt.fill_between(
        iterations,
        upper_bounds,
        lower_bounds,
        color="orange",
        alpha=0.3,
        label="Error Bars",
    )
    # plt.plot(iterations, imps[:cut])
    # plt.title("Training curve")
    plt.xlabel("Epochs")
    plt.ylabel("Imps/deal")
    plt.show()
    # plt.savefig("figs/training_curve.svg", format="svg")
