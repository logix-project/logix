import copy
from typing import Dict

import matplotlib.pyplot as plt
import torch

from experiments.eval_utils import compute_lds, compute_lds_fast, invert_mask
from experiments.plotting_utils import (
    CAP_SIZE,
    MARKER_SIZE,
    get_color,
    get_name,
    init_plotting,
)

GET_RESULTS_FAST = True


def compute_single_model_lso_scores(
    data_name: str, algo_name: str, model_id: int, num_masks: int = 120
) -> Dict[float, Dict[str, float]]:
    alpha_lst = [0.0, 0.1, 0.3, 0.5, 0.7]

    total_results = {}
    for alpha in alpha_lst:
        total_results[alpha] = {}

        lso_scores = torch.load(
            f"../../files/lso_scores/data_{data_name}/{alpha}_5r_50br.pt",
            map_location="cpu",
        )
        trak_diff_scores = -lso_scores["trak_margin_diffs"][:num_masks, :]

        if "prototype" in algo_name:
            algo_scores = torch.load(
                f"../../files/prototype_results/data_{data_name}/model_{model_id}/{algo_name}.pt",
                map_location="cpu",
            )
        elif "unif" in algo_name:
            algo_scores = 0
            for mid in range(9):
                algo_scores += torch.load(
                    f"../../files/ensemble_unif_results/data_{data_name}/model_{mid}/{algo_name}.pt",
                    map_location="cpu",
                )
            algo_scores += torch.load(
                f"../../files/unif_results/data_{data_name}/model_0/{algo_name}.pt",
                map_location="cpu",
            )
        else:
            if "trak" in algo_name:
                algo_scores = torch.load(
                    f"../../files/ensemble_results/data_{data_name}/alpha_0.5/{algo_name}_new_30.pt",
                    map_location="cpu",
                )
            else:
                algo_scores = torch.load(
                    f"../../files/ensemble_results/data_{data_name}/alpha_0.0/{algo_name}.pt",
                    map_location="cpu",
                )
        full_mask = invert_mask(lso_scores["binary_masks"])
        full_mask = full_mask[:, :num_masks]
        algo_preds = (algo_scores @ full_mask).t().numpy()

        if GET_RESULTS_FAST:
            statistics = compute_lds_fast(trak_diff_scores, algo_preds)
        else:
            statistics = compute_lds(trak_diff_scores, algo_preds)

        total_results[alpha] = statistics
    return total_results


def main(data_name: str, model_id: int, num_masks: int) -> None:
    alpha_lst = [0.0, 0.1, 0.3, 0.5, 0.7]

    algo_name_lst = [
        "representation_similarity_dot",
        "tracin_dot",
        # "tracin_cos",
        "trak",
        "if_d1e-08",
        # "unif_average",
        "unif_segment"
    ]
    total_results = {}
    for algo_name in algo_name_lst:
        algo_results = compute_single_model_lso_scores(
            data_name=data_name.lower(),
            algo_name=algo_name,
            num_masks=num_masks,
            model_id=model_id,
        )
        total_results[algo_name] = algo_results

    init_plotting()
    for algo_name in algo_name_lst:
        print(algo_name)
        mean = [total_results[algo_name][alpha]["mean"] for alpha in alpha_lst]
        print(mean)
        if "ensemble" in algo_name:
            plt.plot(
                alpha_lst,
                mean,
                "o--",
                c=get_color(algo_name),
                markersize=MARKER_SIZE,
            )
        else:
            plt.plot(
                alpha_lst,
                mean,
                "o-",
                label=get_name(algo_name),
                c=get_color(algo_name),
                markersize=MARKER_SIZE,
            )

        if not GET_RESULTS_FAST:
            ci = [total_results[algo_name][alpha]["ci"] for alpha in alpha_lst]
            print(ci)
            plt.errorbar(
                alpha_lst,
                mean,
                yerr=ci,
                fmt="o",
                c=get_color(algo_name),
                capsize=CAP_SIZE,
                markersize=MARKER_SIZE,
            )
    plt.legend()
    plt.title(f"Dataset: {data_name} (BERT)")
    plt.ylabel("Correlations")
    plt.xlabel(r"$\alpha$")
    alpha_labels = copy.deepcopy(alpha_lst)
    alpha_labels[0] = r"$1/N$"
    plt.xticks(alpha_lst, alpha_labels)
    # plt.ylim(-0.05, 0.6)
    plt.grid()
    plt.show()
    plt.clf()


if __name__ == "__main__":
    # main(data_name="QNLI", num_masks=100, model_id=0)
    # main(data_name="SST2", num_masks=100, model_id=0)
    main(data_name="RTE", num_masks=100, model_id=0)
