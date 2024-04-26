import math

import matplotlib.pyplot as plt
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
        "gradient_similarity_dot",
        "tracin_dot",
        "if_d1e-08",
        # "unif_average",
        # "unif_segment",
    ]
    num_train = 51_200
    num_corrupt = math.ceil(0.1 * num_train)
    num_interval = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    num_train_interval = [math.ceil(x * num_train) for x in num_interval]

    init_plotting()
    plt.plot(
        num_interval,
        num_interval,
        "o-",
        label=get_name("random"),
        c=get_color("random"),
        markersize=MARKER_SIZE,
    )

    for algo_name in algo_name_lst:
        total_acc_lst = []
        for mid in range(5):
            results = torch.load(
                f"../../files/corruption_results/data_{data_name.lower()}/model_{mid}/{algo_name}.pt",
                map_location="cpu",
            )
            top_idxs = torch.argsort(results, descending=True)

            acc_lst = []
            for nt in num_train_interval:
                acc_lst.append(torch.sum(top_idxs[:nt] <= num_corrupt) / num_corrupt)
            total_acc_lst.append(acc_lst)

        mean = torch.tensor(total_acc_lst).mean(0)
        std = torch.tensor(total_acc_lst).std(0)
        plt.plot(
            num_interval,
            mean,
            "o-",
            label=get_name(algo_name),
            c=get_color(algo_name),
            markersize=MARKER_SIZE,
        )
        plt.errorbar(
            num_interval,
            mean,
            yerr=std,
            fmt="o",
            c=get_color(algo_name),
            capsize=CAP_SIZE,
            markersize=MARKER_SIZE,
        )

    plt.legend(ncol=2)
    plt.title(f"Dataset: {data_name} (BERT)")

    plt.ylabel("Frac. of Corruption Detected")
    plt.xlabel("Fraction of Training Samples Inspected")
    plt.grid()
    plt.show()
    plt.clf()


if __name__ == "__main__":
    main(data_name="sst2")
