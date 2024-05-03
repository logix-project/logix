import time
import argparse

import torch
import os

import copy

from logix import LogIX, LogIXScheduler
from logix.utils import DataIDGenerator
from logix.analysis import InfluenceFunction
from examples.brittleness.pipeline import construct_model, get_hyperparameters, get_loaders, get_eval_train_loader_with_aug

from tqdm import tqdm

from examples.compute_utils import get_ensemble_file_name, get_expt_name_by_config

parser = argparse.ArgumentParser("CIFAR Influence Analysis")
parser.add_argument("--data", type=str, default="cifar10")
parser.add_argument("--damping", type=float, default=None)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--use_augmented_data", action="store_true")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--model_id", type=int, default=0)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = "../files/ensemble_results"

alpha = 0.0
data_name = args.data
os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(f"{BASE_PATH}/data_{data_name}/", exist_ok=True)
os.makedirs(f"{BASE_PATH}/data_{data_name}/alpha_{alpha}/", exist_ok=True)

model = construct_model(name=data_name).to(DEVICE)
hyper_dict = get_hyperparameters(data_name)
final_epoch = hyper_dict["epochs"]

# Get a single checkpoint (first model_id and last epoch).
model.load_state_dict(
    torch.load(f"../files/checkpoints/data_{data_name}/model_{args.model_id}/epoch_{final_epoch}.pt", map_location="cpu")
)
model.eval()

_, eval_train_loader ,valid_loader = get_loaders(
    data_name=data_name,
    eval_batch_size = 256,
)

logix = LogIX(project="test_nosave", config="./config.yaml")
logix_scheduler = logix.scheduler

expt_name = get_expt_name_by_config(logix.config, args)

file_name = get_ensemble_file_name(
    base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, alpha=alpha
)
if os.path.exists(file_name):
    print("File already exists")
    # exit()
else:
    print(f"File to save: {file_name}")
# Gradient & Hessian logging
logix.watch(model)
id_gen = DataIDGenerator()
if not args.resume:
    for epoch in logix_scheduler:
        for inputs, targets in tqdm(eval_train_loader):
            data_id = id_gen(inputs)
            with logix(data_id=data_id):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                model.zero_grad()
                outs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
                loss.backward()
        logix.finalize()
else:
    logix.initialize_from_log()

    # Influence Analysis
print("logix finalized on train set\n")
query_iter = iter(valid_loader)
test_input, test_target = next(query_iter)
test_id = id_gen(test_input)
logix.eval()

with logix(data_id=test_id) as al:
    test_input, test_target = test_input.to(DEVICE), test_target.to(DEVICE)
    model.zero_grad()
    test_out = model(test_input)
    test_loss = torch.nn.functional.cross_entropy(
        test_out, test_target, reduction="sum"
    )
    test_loss.backward()
    test_log = copy.deepcopy(logix.get_log())

if args.use_augmented_data:
    get_loader_fn = get_eval_train_loader_with_aug
else:
    get_loader_fn = get_loaders

_, eval_train_loader ,_ = get_loader_fn(
    data_name=data_name,
    eval_batch_size = 16,
)
if_scores_total = []

if logix.config.scheduler.save == "none":
    for inputs, targets in tqdm(eval_train_loader):
        data_id = id_gen(inputs)
        with logix(data_id=data_id):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            model.zero_grad()
            outs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
            loss.backward()
            train_log = logix.get_log()
        if_score = logix.influence.compute_influence(
            test_log, train_log, damping=args.damping,
            precondition=logix.config.scheduler.hessian != "none"
        )
        if_scores_total.append(if_score['influence'])
    if_scores = torch.cat(if_scores_total, dim=-1)
else:
    log_loader = logix.build_log_dataloader()
    for train_log in tqdm(log_loader):
        if_score = logix.influence.compute_influence(
            test_log, train_log, damping=args.damping,
            precondition=logix.config.scheduler.hessian != "none"
        )
        if_scores_total.append(if_score['influence'])
    if_scores = torch.cat(if_scores_total, dim=-1)

torch.save(if_scores, file_name)