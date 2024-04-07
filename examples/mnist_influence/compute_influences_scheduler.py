import time
import argparse

import torch
from logix import LogIX, LogIXScheduler
from logix.utils import DataIDGenerator
from logix.analysis import InfluenceFunction

from train import (
    get_mnist_dataloader,
    get_fmnist_dataloader,
    construct_mlp,
)

parser = argparse.ArgumentParser("MNIST Influence Analysis")
parser.add_argument("--data", type=str, default="mnist", help="mnist or fmnist")
parser.add_argument("--eval-idxs", type=int, nargs="+", default=[0])
parser.add_argument("--damping", type=float, default=1e-5)
parser.add_argument("--hessian", type=str, default="none")
parser.add_argument("--lora", type=str, default="none")
parser.add_argument("--save", type=str, default="grad")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = construct_mlp().to(DEVICE)
model.load_state_dict(
    torch.load(f"checkpoints/{args.data}_0_epoch_9.pt", map_location="cpu")
)
model.eval()

dataloader_fn = get_mnist_dataloader if args.data == "mnist" else get_fmnist_dataloader
train_loader = dataloader_fn(
    batch_size=512, split="train", shuffle=False, subsample=True
)
query_loader = dataloader_fn(
    batch_size=1, split="valid", shuffle=False, indices=args.eval_idxs
)

logix = LogIX(project="test", config="./config.yaml")
al_scheduler = LogIXScheduler(
    logix, lora=args.lora, hessian=args.hessian, save=args.save
)

# Gradient & Hessian logging
logix.watch(model)
id_gen = DataIDGenerator()
from tqdm import tqdm

if not args.resume:
    for epoch in al_scheduler:
        for inputs, targets in tqdm(train_loader):
            data_id = id_gen(inputs)
            with logix(data_id=data_id):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                model.zero_grad()
                outs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
                loss.backward()
        logix.finalize()
else:
    if args.lora:
        logix.add_lora()
    logix.initialize_from_log()

# Influence Analysis
log_loader = logix.build_log_dataloader(batch_size=64, num_workers=4)

# logix.add_analysis({"influence": InfluenceFunction})
query_iter = iter(query_loader)
test_input, test_target = next(query_iter)
test_id = id_gen(test_input)
logix.setup({"log": "grad"})
logix.eval()
with logix(data_id=test_id):
    test_input, test_target = test_input.to(DEVICE), test_target.to(DEVICE)
    model.zero_grad()
    test_out = model(test_input)
    test_loss = torch.nn.functional.cross_entropy(
        test_out, test_target, reduction="sum"
    )
    test_loss.backward()
test_log = logix.get_log()
if_scores = logix.influence.compute_influence_all(
    test_log, log_loader, damping=args.damping
)
_, top_influential_data = torch.topk(if_scores, k=10)

# Save
if_scores = if_scores.cpu().numpy().tolist()[0]
torch.save(if_scores, "if_logix_scheduler.pt")
print("Top influential data indices:", top_influential_data.cpu().numpy().tolist())
