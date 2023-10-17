import os
from typing import List

import torch

from pipeline import construct_mlp, get_loaders

BASE_PATH = "files/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_everything(data_name: str, model_id: int = 0):
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(f"{BASE_PATH}/{model_id}", exist_ok=True)

    _, eval_train_loader, valid_loader = get_loaders(
        data_name=data_name, train_indices=None, valid_indices=list(range(16))
    )

    model = construct_mlp()
    model.load_state_dict(
        torch.load(
            f"files/checkpoints/{model_id}/{data_name}_epoch_20.pt",
            map_location="cpu",
        )
    )
    model.eval()

    return model.to(DEVICE), eval_train_loader, valid_loader
