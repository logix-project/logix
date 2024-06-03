import time
import argparse

from tqdm import tqdm
import torch

from logix import LogIX, LogIXScheduler
from logix.utils import DataIDGenerator
from logix.analysis import InfluenceFunction

from train import (
    get_cifar10_dataloader,
    construct_rn9,
)

parser = argparse.ArgumentParser("CIFAR Influence Analysis")
parser.add_argument("--data", type=str, default="cifar10", help="cifar10/100")
parser.add_argument("--eval-idxs", type=int, nargs="+", default=[0])
parser.add_argument("--damping", type=float, default=None)
parser.add_argument("--lora", type=str, default="none")
parser.add_argument("--hessian", type=str, default="raw")
parser.add_argument("--save", type=str, default="grad")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = construct_rn9().to(DEVICE)

# Get a single checkpoint (first model_id and last epoch).
model.load_state_dict(
    torch.load(f"checkpoints/{args.data}_0_epoch_23.pt", map_location="cpu")
)
model.eval()

dataloader_fn = get_cifar10_dataloader
train_loader = dataloader_fn(
    batch_size=512, split="train", shuffle=False, subsample=True, augment=False
)
test_loader = dataloader_fn(
    batch_size=1, split="valid", shuffle=False, indices=args.eval_idxs, augment=False
)

logix = LogIX(project="test", config="./config.yaml")
logix_scheduler = LogIXScheduler(
    logix, lora=args.lora, hessian=args.hessian, save=args.save
)

# Gradient & Hessian logging
logix.watch(model)

id_gen = DataIDGenerator()
for epoch in logix_scheduler:
    for inputs, targets in tqdm(train_loader, desc="Extracting log"):
        with logix(data_id=id_gen(inputs)):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            model.zero_grad()
            outs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
            loss.backward()
    logix.finalize()

# Influence Analysis
log_loader = logix.build_log_dataloader()

logix.eval()
logix.setup({"grad": ["log"]})
for test_input, test_target in test_loader:
    with logix(data_id=id_gen(test_input)):
        test_input, test_target = test_input.to(DEVICE), test_target.to(DEVICE)
        model.zero_grad()
        test_out = model(test_input)
        test_loss = torch.nn.functional.cross_entropy(
            test_out, test_target, reduction="sum"
        )
        test_loss.backward()
        test_log = logix.get_log()

    # Influence computation
    result = logix.influence.compute_influence_all(
        test_log, log_loader, damping=args.damping
    )
    break

# Save
if_scores = result["influence"].numpy().tolist()
torch.save(if_scores, "if_logix.pt")
