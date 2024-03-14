import time
import argparse

import torch
import os

import copy

from analog import AnaLog, AnaLogScheduler
from analog.utils import DataIDGenerator
from analog.analysis import InfluenceFunction
from examples.pipeline import construct_model, get_hyperparameters, get_loaders

from tqdm import tqdm

from examples.compute_utils import get_ensemble_file_name, get_expt_name_by_config

parser = argparse.ArgumentParser("CIFAR Influence Analysis")
parser.add_argument("--data", type=str, default="fmnist")
parser.add_argument("--damping", type=float, default=1e-10)
parser.add_argument("--ekfac", action="store_true")
parser.add_argument("--lora", action="store_true")
parser.add_argument("--sample", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--model_id", type=int, default=0)
parser.add_argument("--save", action="store_true")
parser.add_argument("--expt_name_additional_tag", type=str, default="")
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

analog = AnaLog(project="test_nosave", config="./config.yaml")
analog_scheduler = AnaLogScheduler(analog, lora=args.lora, ekfac=args.ekfac, sample=args.sample)
if not args.save:
    analog_scheduler.analog_state_schedule[-1]["save"] = None # hacky way to disable saving

expt_name = get_expt_name_by_config(analog.config, args.lora, args.ekfac, args.model_id, args.damping,"", args.sample)

file_name = get_ensemble_file_name(
    base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, alpha=alpha
)
if os.path.exists(file_name):
    print("File already exists")
    exit()
# Gradient & Hessian logging
analog.watch(model)
id_gen = DataIDGenerator()
if not args.resume:
    for epoch in analog_scheduler:
        sample = True if epoch < (len(analog_scheduler) - 1) and args.sample else False
        for inputs, targets in tqdm(eval_train_loader):
            data_id = id_gen(inputs)
            with analog(data_id=data_id):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                model.zero_grad()
                outs = model(inputs)
                if sample:
                    probs = torch.nn.functional.softmax(outs, dim=-1)
                    targets = torch.multinomial(probs, 1).flatten().detach()
                loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
                loss.backward()
        analog.finalize()
else:
    analog.initialize_from_log()

    # Influence Analysis
print("analog finalized on train set\n")
analog.add_analysis({"influence": InfluenceFunction})
query_iter = iter(valid_loader)
test_input, test_target = next(query_iter)
test_id = id_gen(test_input)
analog.eval()

if args.save:
    log_loader = analog.build_log_dataloader(batch_size = 1)
    test_log = copy.deepcopy(next(log_loader))
else:
    with analog(data_id=test_id) as al:
        test_input, test_target = test_input.to(DEVICE), test_target.to(DEVICE)
        model.zero_grad()
        test_out = model(test_input)
        test_loss = torch.nn.functional.cross_entropy(
            test_out, test_target, reduction="sum"
        )
        test_loss.backward()
        test_log = copy.deepcopy(analog.get_log())

_, eval_train_loader ,_ = get_loaders(
    data_name=data_name,
    eval_batch_size = 16,
)
if_scores_total = []
for inputs, targets in tqdm(eval_train_loader):
    data_id = id_gen(inputs)
    with analog(data_id=data_id):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        model.zero_grad()
        outs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
        loss.backward()
        train_log = analog.get_log()
    if_score = analog.influence.compute_influence(
        test_log, train_log, damping=args.damping
    )
    if_scores_total.append(if_score)
if_scores = torch.cat(if_scores_total, dim=-1)

torch.save(if_scores, file_name)
