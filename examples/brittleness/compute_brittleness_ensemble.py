import argparse
import os.path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from examples.pipeline import (
    construct_model,
    get_hyperparameters,
    get_loaders,
    get_remove_intervals,
)
from examples.utils import save_tensor, set_seed
from analog import AnaLog

from examples.compute_utils import get_ensemble_file_name, get_expt_name_by_config

from analog.analog import Config

BASE_PATH = "../files/ensemble_brittleness_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


def get_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        acc_lst = []
        for images, labels in loader:
            images, labels = images.to(device=DEVICE), labels.to(device=DEVICE)
            outputs = model(images)
            accs = (outputs.argmax(-1) == labels).float().cpu()
            acc_lst.append(accs)
        all_accs = torch.cat(acc_lst)
    return all_accs


def train_with_indices(
    data_name: str, model_id: int, idxs_to_keep: Optional[List[int]] = None
) -> nn.Module:
    hyper_dict = get_hyperparameters(data_name)
    print(hyper_dict)
    lr = hyper_dict["lr"]
    wd = hyper_dict["wd"]
    epochs = int(hyper_dict["epochs"])

    if idxs_to_keep is None:
        train_loader, _, _ = get_loaders(data_name=data_name)
    else:
        train_loader, _, _ = get_loaders(
            data_name=data_name,
            train_indices=idxs_to_keep,
        )
    set_seed(model_id + 1234)
    model = construct_model(name=data_name).to(device=DEVICE)
    model = train(
        model=model, loader=train_loader, lr=lr, weight_decay=wd, epochs=epochs
    )
    return model


def train_with_configurations(
    data_name: str,
    num_train: int,
    mask: torch.Tensor,
    top_idxs: List[int],
    valid_loader: torch.utils.data.DataLoader,
    intervals: List[int],
    seed_ids: List[int],
):
    assert len(top_idxs) == num_train
    mean_lst = []
    std_lst = []
    valid_total_raw_acc_lst = []
    for ri in intervals:
        idxs_to_remove = top_idxs[:ri]
        idxs_to_keep = list(set(range(num_train)) - set(idxs_to_remove))
        assert len(idxs_to_keep) + len(idxs_to_remove) == num_train

        valid_acc_lst = []
        raw_valid_acc_lst = []
        for seed in seed_ids:
            model = train_with_indices(
                data_name=data_name, idxs_to_keep=idxs_to_keep, model_id=seed
            )
            valid_results = get_accuracy(model, valid_loader)
            valid_acc_lst.append(valid_results[mask].sum())
            raw_valid_acc_lst.append(valid_results)

        mean_lst.append(torch.mean(torch.stack(valid_acc_lst)).item())
        std_lst.append(torch.std(torch.stack(valid_acc_lst)).item())
        valid_total_raw_acc_lst.append(raw_valid_acc_lst)

    results_dict = {
        "intervals": intervals,
        "mean": mean_lst,
        "std": std_lst,
        "raw_lst": valid_total_raw_acc_lst,
    }
    return results_dict


def get_file_name(expt_name: str, data_name: str) -> str:
    return f"{BASE_PATH}/data_{data_name}/{expt_name}.pt"


def main(data_name: str, algo_name_lst: List[str], startIdx, endIdx) -> None:
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(f"{BASE_PATH}/data_{data_name.lower()}/", exist_ok=True)

    valid_target_num = 100
    _, eval_train_loader, valid_loader = get_loaders(
        data_name=data_name.lower(),
        valid_indices=list(range(valid_target_num)),
    )
    num_train = len(eval_train_loader.dataset)

    seed_ids = list(range(3))
    remove_intervals = get_remove_intervals(data_name)
    expt_name = "base"
    file_name = get_file_name(expt_name=expt_name, data_name=data_name.lower())
    if os.path.exists(file_name):
        print(f"Found existing results at {file_name}.")
        base_results = torch.load(file_name)
    else:
        valid_raw_acc_lst = []
        valid_acc_lst = []
        for seed in seed_ids:
            model = train_with_indices(
                data_name=data_name.lower(), idxs_to_keep=None, model_id=seed
            )
            valid_results = get_accuracy(model, valid_loader)
            valid_acc_lst.append(valid_results.sum())
            valid_raw_acc_lst.append(valid_results)

        base_results = {
            "mean": torch.mean(torch.stack(valid_acc_lst)).item(),
            "std": torch.std(torch.stack(valid_acc_lst)).item(),
            "div_factor": torch.from_numpy(
                (np.array(valid_raw_acc_lst).mean(0) >= 0.5)
            ).sum(),
            "mask": torch.from_numpy((np.array(valid_raw_acc_lst).mean(0) >= 0.5)),
        }
        save_tensor(tensor=base_results, file_name=file_name, overwrite=False)
    mask = base_results["mask"]
    print(mask)

    expt_name = "random"
    print(expt_name)
    file_name = get_file_name(expt_name=expt_name, data_name=data_name.lower())
    file_name = f"{file_name[:-3]}_{startIdx}_{endIdx}.pt"
    if os.path.exists(file_name):
        print(f"Found existing results at {file_name}.")
    else:
        total_success_lst = []
        for i in range(startIdx, endIdx):
            print(f"{i}th validation data point.")
            if mask[i]:
                random_idxs = list(np.random.permutation(list(range(num_train))))
                results = train_with_configurations(
                    data_name=data_name.lower(),
                    num_train=num_train,
                    mask=mask,
                    top_idxs=random_idxs,
                    valid_loader=valid_loader,
                    intervals=remove_intervals,
                    seed_ids=seed_ids,
                )
                success_lst = []
                for j, ri in enumerate(remove_intervals):
                    if np.array(results["raw_lst"][j]).mean(0)[i] < 0.5:
                        success_lst.append(1)
                    else:
                        success_lst.append(0)
                total_success_lst.append(success_lst)
        results = {"results": total_success_lst}
        save_tensor(tensor=results, file_name=file_name, overwrite=False)

    for algo_name in algo_name_lst:
        print(algo_name)
        expt_name = algo_name
        file_name = get_file_name(expt_name=expt_name, data_name=data_name.lower())
        file_name = f"{file_name[:-3]}_{startIdx}_{endIdx}.pt"
        if os.path.exists(file_name):
            print(f"Found existing results at {file_name}.")
        else:
            if "prototype" in algo_name:
                # algo_scores = torch.load(
                #     f"../files/prototype_results/data_{data_name.lower()}/model_{model_id}/{algo_name}.pt",
                #     map_location="cpu",
                # )
                raise NotImplementedError()
            elif "unif" in algo_name:
                # algo_scores = torch.load(
                #     f"../files/unif_results/data_{data_name.lower()}/model_{model_id}/{algo_name}.pt",
                #     map_location="cpu",
                # )
                raise NotImplementedError()
            else:
                ensemble_alpha = "0.5" if "trak" in algo_name else "0.0"
                algo_scores = torch.load(
                    f"../files/ensemble_results/data_{data_name}/alpha_{ensemble_alpha}/{algo_name}.pt",
                    map_location="cpu",
                )
            total_success_lst = []
            for i in range(startIdx, endIdx):
                print(f"{i}th validation data point.")
                if mask[i]:
                    top_idxs = torch.argsort(algo_scores[i], descending=True)
                    top_idxs = [ti.item() for ti in top_idxs]

                    results = train_with_configurations(
                        data_name=data_name.lower(),
                        num_train=num_train,
                        mask=mask,
                        top_idxs=top_idxs,
                        valid_loader=valid_loader,
                        intervals=remove_intervals,
                        seed_ids=seed_ids,
                    )
                    success_lst = []
                    for j, ri in enumerate(remove_intervals):
                        if np.array(results["raw_lst"][j]).mean(0)[i] < 0.5:
                            success_lst.append(1)
                        else:
                            success_lst.append(0)
                    total_success_lst.append(success_lst)
            results = {"results": total_success_lst}
            save_tensor(tensor=results, file_name=file_name, overwrite=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CIFAR Influence Analysis")
    parser.add_argument("--startIdx", type=int, default=0)
    parser.add_argument("--endIdx", type=int, default=10)
    parser.add_argument("--scoreFileName", type=str)

    parser.add_argument("--data", type=str, default="fmnist")
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use_augmented_data", action="store_true")
    parser.add_argument("--grad_sim", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--model_id", type=int, default=0, help="10 for 10 ensemble models, 0-9 for single model")

    algo_name_lst = []
    args = parser.parse_args()
    if args.scoreFileName is not None:
        algo_name_lst.append(args.scoreFileName)
        print(f"scoreFileName is set to {args.scoreFileName}, ignore other options")
    else:
        analog = AnaLog(project="brittleness", config="./config.yaml")
        scoreFileName = get_expt_name_by_config(analog.config, args)
        algo_name_lst.append(scoreFileName)

    if args.data == "mnist" or args.data == "fmnist":
        from examples.mnist_influence.train import train
    elif args.data == "cifar10":
        from examples.cifar_influence.train import train
    else:
        raise NotImplementedError()

    main(
        data_name=args.data,
        algo_name_lst=algo_name_lst,
        startIdx=args.startIdx,
        endIdx=args.endIdx,
    )
