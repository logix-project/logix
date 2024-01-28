import os

import matplotlib.pyplot as plt
from tueplots import bundles, markers

CAP_SIZE = 2
MARKER_SIZE = 4

# Colors are taken from:
# https://colorbrewer2.org/.
ALGORITHMS_TO_COLOR = {
    "random": "#a6cee3",
    "representation_similarity_dot": "#1f78b4",
    "representation_similarity_cos": "#1f78b4",
    "gradient_similarity_dot": "#fdbf6f",
    "gradient_similarity_cos": "#fdbf6f",
    "tracin_dot": "#b2df8a",
    "tracin_cos": "#33a02c",
    "trak": "#ff7f00",
    "trak_ensemble": "#ff7f00",
}


ALGORITHMS_TO_NAME = {
    "random": "Random",
    "representation_similarity_dot": "RepSim",
    "representation_similarity_cos": "RepSim",
    "gradient_similarity_dot": "GradSim",
    "gradient_similarity_cos": "GradSim",
    "tracin_dot": "TracIn",
    "tracin_cos": "GAS",
    "trak": "Trak",
    "trak_ensemble": "Trak",
}


def get_color(algo_name: str) -> str:
    if algo_name in ALGORITHMS_TO_COLOR.keys():
        return ALGORITHMS_TO_COLOR[algo_name]
    elif "if_d" in algo_name:
        return "#e31a1c"
    elif (
        "prototype_if_average" in algo_name
        or "prototype_adam_if_average" in algo_name
        or "unif_average" in algo_name
    ):
        return "#fb9a99"
    elif (
        "prototype_if_segment" in algo_name
        or "prototype_if_normalized_segment" in algo_name
        or "prototype_adam_if_segment" in algo_name
        or "prototype_if_new_segment" in algo_name
        or "unif_segment" in algo_name
    ):
        return "#cab2d6"
    elif "pca" in algo_name:
        return "#6a3d9a"
    elif "noLora" in algo_name:
        return "#ffff99"
    else:
        raise NotImplementedError()


def get_name(algo_name: str) -> str:
    if algo_name in ALGORITHMS_TO_NAME.keys():
        return ALGORITHMS_TO_NAME[algo_name]
    elif "if_d" in algo_name:
        return "IF"
    elif "prototype_if_tracin" in algo_name:
        return "IF TracIn"
    elif (
        "prototype_if_average" in algo_name or "prototype_adam_if_average" in algo_name
    ):
        return r"UNIF ($L=1$)"
    elif "unif_average" in algo_name:
        return r"UNIF ($L=1$)"
    elif (
        "prototype_if_segment" in algo_name
        or "prototype_if_normalized_segment" in algo_name
        or "prototype_adam_if_segment" in algo_name
        or "prototype_if_new_segment" in algo_name
    ):
        return r"IF Segment"
    elif "unif_segment" in algo_name:
        return r"UNIF ($L=3$)"
    elif "pca" in algo_name:
        return algo_name.upper()
    elif "noLora" in algo_name:
        return algo_name.upper()
    else:
        raise NotImplementedError()


def init_plotting(
    column: str = "half", nrows: int = 1, ncols: int = 1, usetex: bool = True
) -> None:
    os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
    plt.rcParams.update(
        bundles.icml2022(
            column=column, nrows=nrows, ncols=ncols, usetex=usetex
        )
    )
    plt.rcParams.update(markers.with_edge())
    plt.rcParams.update({"figure.dpi": 600})
