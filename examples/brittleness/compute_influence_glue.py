import argparse

import torch
import os

import copy

import logix
from examples.brittleness.pipeline import construct_model, get_hyperparameters, get_loaders

from tqdm import tqdm

from examples.compute_utils import get_ensemble_file_name, get_expt_name_by_config

from transformers import AutoTokenizer

parser = argparse.ArgumentParser("CIFAR Influence Analysis")
parser.add_argument("--data", type=str, default="rte")
parser.add_argument("--damping", type=float, default=None)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--model_id", type=int, default=0)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
BASE_PATH = "../files/ensemble_results"

alpha = 0.0
TEST_SAMPLE = 200
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
    eval_batch_size = 16, # use 200 valid samples
)

# Set-up LogIX
run = logix.init("test_nosave", config="./config.yaml")

logix_scheduler = run.scheduler

expt_name = get_expt_name_by_config(run.config, args)

file_name = get_ensemble_file_name(
    base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, alpha=alpha
)
if os.path.exists(file_name):
    print("File already exists")
    exit()
else:
    print(f"File to save: {file_name}")
# Gradient & Hessian logging
logix.watch(model)

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-cased", use_fast=True, trust_remote_code=True
)

logix.eval()

if not args.resume:
    for epoch in logix_scheduler:
        for batch in tqdm(eval_train_loader):
            data_id = tokenizer.batch_decode(batch["input_ids"])
            labels = batch.pop("labels").view(-1).to(device=DEVICE)
            _ = batch.pop("idx")  # Ignored index.

            with run(data_id=data_id):
                model.zero_grad()
                outputs = model(
                    batch["input_ids"].to(device=DEVICE),
                    batch["token_type_ids"].to(device=DEVICE),
                    batch["attention_mask"].to(device=DEVICE),
                )
                logits = outputs.view(-1, outputs.shape[-1])
                loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                loss.backward()
        logix.finalize()
else:
    logix.initialize_from_log()


if_scores_total_test = []
test_processed = 0
for test_batch in valid_loader:
    # Influence Analysis
    if test_processed > 200:
        break
    print("logix finalized on train set\n")
    test_id = tokenizer.batch_decode(test_batch["input_ids"])
    logix.eval()

    with run(data_id=test_id):
        model.zero_grad()
        test_out = model(
            test_batch["input_ids"].to(device=DEVICE),
            test_batch["token_type_ids"].to(device=DEVICE),
            test_batch["attention_mask"].to(device=DEVICE),
        )
        test_target = test_batch["labels"].to(device=DEVICE)
        test_loss = torch.nn.functional.cross_entropy(
            test_out, test_target, reduction="sum"
        )
        test_loss.backward()
        test_log = copy.deepcopy(logix.get_log())

    _, eval_train_loader ,_ = get_loaders(
        data_name=data_name,
        eval_batch_size = 16,
    )
    if_scores_total_train = []

    if run.config.scheduler.save == "none":
        for batch in tqdm(eval_train_loader):
            # Determine if this is an RTE task and generate the data_id accordingly.
            data_id = tokenizer.batch_decode(batch["input_ids"])
            labels = batch.pop("labels").view(-1).to(device=DEVICE)
            mask = batch["attention_mask"].to(device=DEVICE)
            # For RTE, we also need token_type_ids.
            token_type_ids = batch["token_type_ids"].to(device=DEVICE)
            input_ids = batch["input_ids"].to(device=DEVICE)

            with run(data_id=data_id):
                model.zero_grad()
                outs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=mask)
                loss = torch.nn.functional.cross_entropy(outs, labels, reduction="sum")
                loss.backward()
                train_log = run.get_log()
                
            # Compute influence.
            if_score = run.influence.compute_influence(
                test_log, train_log, damping=args.damping,
                precondition = run.config.scheduler.hessian != "none"
            )
            if_scores_total_train.append(if_score['influence'])
        if_scores = torch.cat(if_scores_total_train, dim=-1)
    else:
        log_loader = run.build_log_dataloader()
        for train_log in tqdm(log_loader):
            if_score = run.influence.compute_influence(
                test_log, train_log, damping=args.damping,
                precondition = run.config.scheduler.hessian != "none"
            )
            if_scores_total_train.append(if_score['influence'])
        if_scores = torch.cat(if_scores_total_train, dim=-1)
    test_processed += if_scores.shape[0]
    if_scores_total_test.append(if_scores)

if_scores_total_test_stacked = torch.vstack(if_scores_total_test)
torch.save(if_scores_total_test_stacked, file_name)
