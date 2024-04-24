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
    elif algo_name == "pca1e-06":
        return "#6a3d9a"
    elif algo_name == "noLoraEkfac":
        return "#ff7f00"
    elif algo_name == "pca0.0001":
        return "#33a02c"
    elif algo_name == "loraEkfacrandom64":
        return "#b15928"
    elif algo_name == "loraEkfacorthogonal64":
        return "#1f78b4"
    elif algo_name == "loraEkfacpca64":
        return "#e31a1c"
    elif algo_name == "loraEkfacrandom64_damping0.001":
        return "#fdbf6f"
    elif algo_name == "loraEkfacrandom64_damping1e-05":
        return "#ff7f00"
    elif algo_name == "loraEkfacrandom16":
        return "#abdda4"
    elif algo_name == "noLoraEkfac_10_true_fisher":
        return "#001234"
    elif algo_name == "noLoraEkfac_10_aug_trueFisher":
        return "#8dd3c7"
    elif algo_name == "noLoraEkfac_10_aug_empiricalFisher":
        return "#e31a1c"
    elif algo_name == "lora64randomKfac_full_covariance_10":
        return "#b2df8a"
    elif algo_name == "lora64random_full_covariance_aug_10":
        #red
        return "#e31a1c"
    elif algo_name == "initrandom_rank64_use_loraTrue_use_full_covarianceTrue_sampleFalse_ekfacFalse_damping1e-10_use_augmented_dataFalse_grad_simFalse_tag_model_id10":
        # blue
        return "#1f78b4"
    elif algo_name == "initrandom_rank64_use_loraTrue_use_full_covarianceTrue_sampleFalse_ekfacFalse_damping1e-10_use_augmented_dataFalse_grad_simFalse_tag_model_id_10":
        # yellow
        return "#ff7f00"
    elif algo_name == "initrandom_rank64_use_loraTrue_use_full_covarianceTrue_sampleFalse_ekfacFalse_damping1e-10_use_augmented_dataFalse_grad_simFalse_tag_model_id0":
        return "#6704FD"
    elif algo_name == "initrandom_rank64_use_loraFalse_use_full_covarianceFalse_sampleFalse_ekfacFalse_damping1e-10_use_augmented_dataFalse_grad_simFalse_tag_model_id0":
        return "#FF0000"
    elif algo_name == "initrandom_rank64_use_loraTrue_use_full_covarianceTrue_sampleFalse_ekfacFalse_damping1e-10_use_augmented_dataFalse_grad_simFalse_tag_model_id0":
        return "#00FF00"
    elif algo_name == "trak_30":
        return "#a6cee3"
    elif algo_name == "trak_30_model0":
        return "#1f78b4"
    elif algo_name == "Trak_30_model0_epoch25_our_ckpt":
        return "#e31a1c"
    elif algo_name == "initrandom_rank64_use_loraTrue_use_full_covarianceFalse_sampleFalse_ekfacTrue_damping1e-10_use_augmented_dataFalse_grad_simFalse_tag_model_id0":
        return "#729c9c"
    elif algo_name == "initrandom_rank64_use_loraTrue_ekfacTrue_damping1e-10_tag_model_id0_logit_lossTrue":
        return "#b2df8a"
    elif algo_name == "trak_30_single":
        return "#1f78b4"
    elif algo_name == "initrandom_rank64_damping1e-10_tag_model_id0":
        return "#ff7f00"
    elif algo_name == "initrandom_rank64_use_loraTrue_ekfacTrue_damping1e-10_tag_model_id0":
        return "#729c9c"
    elif algo_name == "initrandom_rank64_use_loraTrue_use_full_covarianceTrue_dampingNone_tag_model_id0":
        return "#e31a1c"
    else:
        return "#000000"


def get_name(algo_name: str) -> str:
    if algo_name == "initrandom_rank64_use_loraTrue_use_full_covarianceTrue_sampleFalse_ekfacFalse_damping1e-10_use_augmented_dataFalse_grad_simFalse_tag_model_id_10":
        return "loraEnsembleFullCov"
    elif algo_name == "initrandom_rank64_use_loraFalse_use_full_covarianceFalse_sampleFalse_ekfacFalse_damping1e-10_use_augmented_dataFalse_grad_simFalse_tag_model_id0":
        return "noLoraSingle"
    elif algo_name ==  "initrandom_rank64_use_loraTrue_use_full_covarianceTrue_sampleFalse_ekfacFalse_damping1e-10_use_augmented_dataFalse_grad_simFalse_tag_model_id0":
        return "loraSingleFullCovDamp1e-10"
    elif algo_name == "trak_30_model0":
        return "trakSingle3Epochs"
    elif algo_name == "Trak_30_model0_epoch25_our_ckpt":
        return "trakSingleLastEpoch"
    elif algo_name == "trak_30":
        return "trakEnsemble"
    elif algo_name == "initrandom_rank64_use_loraTrue_use_full_covarianceFalse_sampleFalse_ekfacTrue_damping1e-10_use_augmented_dataFalse_grad_simFalse_tag_model_id0":
        return "loraSingleEkfac"
    elif algo_name == "initrandom_rank64_use_loraTrue_ekfacTrue_damping1e-10_tag_model_id0_logit_lossTrue":
        return "logitLoss"
    elif algo_name == "trak_30_single":
        return "trakSingle"
    elif algo_name == "initrandom_rank64_damping1e-10_tag_model_id0":
        return "noLoraSingle"
    elif algo_name == "initrandom_rank64_use_loraTrue_ekfacTrue_damping1e-10_tag_model_id0":
        return "loraSingleEkfac"
    elif algo_name == "initrandom_rank64_use_loraTrue_use_full_covarianceTrue_dampingNone_tag_model_id0":
        return "loraSingleFullCovDampNone"
    else:
        return algo_name
         


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
