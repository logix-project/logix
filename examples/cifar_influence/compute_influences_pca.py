import time
import argparse

import torch
import os
from analog import AnaLog, AnaLogScheduler
from analog.utils import DataIDGenerator
from analog.analysis import InfluenceFunction
from tqdm import tqdm

from train import (
    get_loaders,
    construct_model,
)
from examples.compute_utils import get_ensemble_file_name

parser = argparse.ArgumentParser("CIFAR Influence Analysis")
parser.add_argument("--data", type=str, default="cifar10", help="cifar10/100")
parser.add_argument("--eval-idxs", type=int, nargs="+", default=[0])
parser.add_argument("--damping", type=float, default=1e-10)
parser.add_argument("--ekfac", action="store_true")
parser.add_argument("--lora", action="store_true")
parser.add_argument("--sample", action="store_true")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = "../files/ensemble_results"

alpha = 0.0
data_name = args.data
os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(f"{BASE_PATH}/data_{data_name}/", exist_ok=True)
os.makedirs(f"{BASE_PATH}/data_{data_name}/alpha_{alpha}/", exist_ok=True)

for model_id in range(10):
    score_table = 0.0
    model = construct_model(data_name="cifar10").to(DEVICE)
    final_epoch = 25

    # Get a single checkpoint (first model_id and last epoch).
    model.load_state_dict(
        torch.load(f"../files/checkpoints/data_{data_name}/model_{model_id}/epoch_{final_epoch}.pt", map_location="cpu")
    )
    model.eval()

    _, eval_train_loader ,valid_loader = get_loaders(
        data_name="cifar10",
        eval_batch_size = 256,
    )

    analog = AnaLog(project="test1", config="./config1.yaml")
    analog_scheduler = AnaLogScheduler(analog, lora=args.lora)

    # Gradient & Hessian logging
    analog.watch(model)
    id_gen = DataIDGenerator()
    if not args.resume:
        for epoch in analog_scheduler:
            for inputs, targets in tqdm(eval_train_loader):
                data_id = id_gen(inputs)
                with analog(data_id=data_id):
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    model.zero_grad()
                    outs = model(inputs)
                    loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
                    loss.backward()
            analog.finalize()
    else:
        analog.initialize_from_log()

        # Influence Analysis
    log_loader = analog.build_log_dataloader()
    print("analog finalized on train set\n")

    analog.add_analysis({"influence": InfluenceFunction})
    query_iter = iter(valid_loader)
    test_input, test_target = next(query_iter)
    test_id = id_gen(test_input)
    analog.eval()
    with analog(data_id=test_id) as al:
        test_input, test_target = test_input.to(DEVICE), test_target.to(DEVICE)
        model.zero_grad()
        test_out = model(test_input)
        test_loss = torch.nn.functional.cross_entropy(
            test_out, test_target, reduction="sum"
        )
        test_loss.backward()
        test_log = analog.get_log()
    start = time.time()
    if_scores = analog.influence.compute_influence_all(
        test_log, log_loader, damping=args.damping
    )
    # Save
    # if_scores = if_scores.numpy().tolist()
    print("Computation time:", time.time() - start)
    score_table += if_scores

score_table /= 10
expt_name = analog.config._lora_config["init"] + str(analog.config._lora_config["compression_ratio_by_memory"])
file_name = get_ensemble_file_name(
    base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, alpha=alpha
)
torch.save(score_table, file_name)