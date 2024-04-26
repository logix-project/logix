import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.plotting_utils import (
    CAP_SIZE,
    MARKER_SIZE,
    get_color,
    get_name,
    init_plotting,
)


def main(data_name: str) -> None:
    algo_name_lst = [
        "random",
        # "representation_similarity_dot",
        "tracin_dot",
        # "tracin_cos",
        # "trak_new",
        "if_d1e-08",
        # "unif_average",
        # "unif_segment",
    ]

    base = 0.0
    intervals = [20, 40, 60, 80, 100, 120]
    init_plotting()

    for algo_name in algo_name_lst:
        print(algo_name)

        if algo_name == "random":
            results = torch.load(
                f"../../files/brittleness_results_v2/data_{data_name.lower()}/{algo_name}.pt"
            )
        else:
            results = torch.load(
                f"../../files/ensemble_brittleness_results_v2/data_{data_name.lower()}/{algo_name}.pt"
            )

        results = np.array(results["results"])

        for i in range(results.shape[0]):
            if results[i].sum() == 0:
                pass
            else:
                found = False
                for j in range(len(results[i])):
                    if results[i][j] == 1:
                        found = True

                    if found:
                        results[i][j] = 1


        avg_results = results.sum(0) / results.shape[0]
        print([base] + list(avg_results))
        plt.plot(
            [0] + intervals,
            [base] + list(avg_results),
            "o-",
            label=get_name(algo_name),
            c=get_color(algo_name),
            markersize=MARKER_SIZE,
        )

    plt.legend(ncol=2)
    plt.title("Dataset: RTE (BERT)")

    plt.ylabel("Frac. of Misclassified Test Examples")
    plt.xlabel("Number of Training Samples Removed")
    plt.grid()
    plt.show()
    plt.clf()


if __name__ == "__main__":
    main(data_name="rte")
