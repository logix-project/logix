import os
import time
import argparse
import yaml

from tqdm import tqdm

import torch
import torch.nn.functional as F
from analog import AnaLog, AnaLogScheduler
from analog.analysis import InfluenceFunction
from analog.utils import DataIDGenerator
from tqdm import tqdm

from utils import construct_model, get_loaders, set_seed

parser = argparse.ArgumentParser("GLUE Influence Analysis")
parser.add_argument("--data_name", type=str, default="wiki")
parser.add_argument("--num_train_data", type=int, default=None)
parser.add_argument("--num_test_data", type=int, default=None)
parser.add_argument("--damping", type=float, default=None)
parser.add_argument("--ekfac", action="store_true")
parser.add_argument("--lora", action="store_true")
parser.add_argument("--sample", action="store_true")
parser.add_argument("--config", type=str, default="lora_pca_32")
parser.add_argument("--project_name", type=str, default=None)
parser.add_argument("--accelerate", action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0)

# model
model = construct_model()
model_path = f"files/checkpoints/0/{args.data_name}_epoch_3.pt"
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.to(DEVICE)
model.eval()

# data
if args.num_test_data is None:
    valid_indices = None
elif isinstance(args.num_test_data, int):
    valid_indices = list(range(args.num_test_data))
if args.num_train_data is not None:
    train_indices = list(range(args.num_train_data))
else:
    train_indices = None
_, eval_train_loader, test_loader = get_loaders(
    train_indices=train_indices,
    valid_indices=valid_indices,
)
num_train = len(eval_train_loader.dataset)
num_test = len(test_loader.dataset)

if args.accelerate:
    from accelerate import Accelerator
    accelerator = Accelerator()
    model, eval_train_loader = accelerator.prepare(model, eval_train_loader)

# Set-up
config_path = f"files/configs/{args.config}.yaml"
assert os.path.exists(config_path), f"Config file {config_path} does not exist"
if args.project_name is None:
    project_name = args.config
else:
    project_name = args.project_name
analog = AnaLog(project=project_name, config=config_path)
config = analog.config.data
al_scheduler = AnaLogScheduler(
    analog, ekfac=args.ekfac, lora=args.lora, sample=args.sample
)

print(f"Experimentting with: {config}")
print(f"Args: {args}")
print(f"Num train: {num_train}, Num test: {num_test}")

# Hessian logging
modules_to_watch = []
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        if "lm_head" in n:
            continue
        modules_to_watch.append(n)
analog.watch(model, name_filter=modules_to_watch)
id_gen = DataIDGenerator(mode="index")
for epoch in al_scheduler:
    # TODO: fix this
    sample = True if epoch < (len(al_scheduler) - 1) and args.sample else False
    for batch in tqdm(eval_train_loader, desc="Hessian logging"):
        data_id = id_gen(batch["input_ids"])
        inputs = (
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        )
        with analog(data_id=data_id, mask=inputs[-1]):
            model.zero_grad()
            lm_logits = model(*inputs)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            if sample:
                with torch.no_grad():
                    reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    probs = torch.nn.functional.softmax(reshaped_shift_logits, dim=-1)
                    sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
                loss = F.cross_entropy(
                    reshaped_shift_logits,
                    sampled_labels.detach(),
                    reduction="sum",
                    ignore_index=-100,
                )
            else:
                labels = batch["labels"].to(DEVICE)
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                    ignore_index=-100,
                )
            loss.backward()
    analog.finalize()

# Compute influence
log_loader = analog.build_log_dataloader()
analog.add_analysis({"influence": InfluenceFunction})
if_scores_list = []
for test_batch in tqdm(test_loader, desc="Computing Influence"):
    with analog(log=["grad"], test=True) as al:
        test_inputs = (
            test_batch["input_ids"].to(DEVICE),
            test_batch["attention_mask"].to(DEVICE),
        )
        test_target = test_batch["labels"].to(DEVICE)
        model.zero_grad()
        test_lm_logits = model(*test_inputs)
        test_shift_logits = test_lm_logits[..., :-1, :].contiguous()

        test_labels = test_batch["labels"].to(DEVICE)
        test_shift_labels = test_labels[..., 1:].contiguous()
        test_loss = F.cross_entropy(
            test_shift_logits.view(-1, test_shift_logits.size(-1)),
            test_shift_labels.view(-1),
            reduction="sum",
            ignore_index=-100,
        )
        test_loss.backward()

        test_log = al.get_log()
        if_scores = analog.influence.compute_influence_all(
            test_log, log_loader, damping=args.damping
        )
        if_scores_list.append(if_scores)
if_scores = torch.cat(if_scores_list, dim=0)

# Save influence scores
log_dir = analog.config.get_storage_config()["log_dir"]
save_path = f"{log_dir}/if_analog.pt"
torch.save(if_scores, save_path)
print(f"Saved influence scores of size {if_scores.size()} to {save_path}")

# Save config
config["misc"] = {}
to_log = ["model_path", "num_train", "num_test", "config_path", "project_name"]
for log in to_log:
    config["misc"][log] = globals()[log]
config["args"] = {}
for k, v in vars(args).items():
    config["args"][k] = v
yaml_path = f"{log_dir}/if_analog.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(config, f)
print(f"Saved config to {yaml_path}")
