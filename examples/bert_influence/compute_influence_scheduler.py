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
parser.add_argument("--data_name", type=str, default="sst2")
parser.add_argument("--num_train_data", type=int, default=None)
parser.add_argument("--num_test_data", type=int, default=None)
parser.add_argument("--damping", type=float, default=1e-5)
parser.add_argument("--ekfac", action="store_true")
parser.add_argument("--lora", action="store_true")
parser.add_argument("--sample", action="store_true")
parser.add_argument("--config", type=str, default="lora_pca_32")
parser.add_argument("--project_name", type=str, default=None)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0)

# model
model = construct_model(args.data_name)
model_path = f"files/checkpoints/0/{args.data_name}_epoch_3.pt"
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.to(DEVICE)
model.eval()

# data
if args.num_test_data is None:
    valid_indices = list(range(32))
elif isinstance(args.num_test_data, int):
    valid_indices = list(range(args.num_test_data))
if args.num_train_data is not None:
    train_indices = list(range(args.num_train_data))
else:
    train_indices = None
_, eval_train_loader, test_loader = get_loaders(
    data_name=args.data_name,
    train_indices=train_indices,
    valid_indices=valid_indices,
)
num_train = len(eval_train_loader.dataset)
num_test = len(test_loader.dataset)
print(f"Train size: {num_train}")
print(f"Test size: {num_test}")

# Set-up
config_path = f"files/configs/{args.config}.yaml"
assert os.path.exists(config_path), f"Config file {config_path} does not exist"
if args.project_name is None:
    project_name = args.config
else:
    project_name = args.project_name
analog = AnaLog(project=project_name, config=config_path)
config = analog.config.data
print(config)
al_scheduler = AnaLogScheduler(
    analog, ekfac=args.ekfac, lora=args.lora, sample=args.sample
)

# Hessian logging
analog.watch(model)
id_gen = DataIDGenerator(mode="index")
for epoch in al_scheduler:
    # TODO: fix this
    sample = True if epoch < (len(al_scheduler) - 1) and args.sample else False
    for batch in tqdm(eval_train_loader, desc="Hessian logging"):
        data_id = id_gen(batch["input_ids"])
        inputs = (
            batch["input_ids"].to(DEVICE),
            batch["token_type_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        )
        with analog(data_id=data_id, mask=inputs[-1]):
            model.zero_grad()
            outputs = model(*inputs)

            if sample:
                raise NotImplementedError

            logits = outputs.view(-1, outputs.shape[-1])
            labels = batch["labels"].view(-1).to(DEVICE)
            loss = F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)
            loss.backward()
    analog.finalize()

# Compute influence
log_loader = analog.build_log_dataloader()
analog.add_analysis({"influence": InfluenceFunction})
test_iter = iter(test_loader)
with analog(log=["grad"], test=True) as al:
    test_batch = next(test_iter)
    test_inputs = (
        test_batch["input_ids"].to(DEVICE),
        test_batch["token_type_ids"].to(DEVICE),
        test_batch["attention_mask"].to(DEVICE),
    )
    test_target = test_batch["labels"].to(DEVICE)
    model.zero_grad()
    test_outputs = model(*test_inputs)

    test_logits = test_outputs.view(-1, test_outputs.shape[-1])
    test_labels = test_batch["labels"].view(-1).to(DEVICE)
    test_loss = F.cross_entropy(
        test_logits,
        test_labels,
        reduction="sum",
        ignore_index=-100,
    )
    test_loss.backward()

    test_log = al.get_log()

start = time.time()
if_scores = analog.influence.compute_influence_all(test_log, log_loader)
print("Computation time:", time.time() - start)

# Save
log_dir = analog.config.get_storage_config()["log_dir"]
config["model_path"] = model_path
save_path = f"{log_dir}/if_analog.pt"
torch.save(if_scores, save_path)
print(f"Saved influence scores of size {if_scores.size()} to {save_path}")
yaml_path = f"{log_dir}/if_analog.yaml"
config["model_path"] = model_path
config["num_train"] = num_train
config["num_test"] = num_test
with open(yaml_path, "w") as f:
    yaml.dump(config, f)
print(f"Saved config to {yaml_path}")
