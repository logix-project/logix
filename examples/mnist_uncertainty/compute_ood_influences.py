import time
import argparse

import tqdm
import numpy as np
import torch

from logix import LogIX
from logix.utils import DataIDGenerator
from logix.analysis import InfluenceFunction
from examples.mnist_influence.utils import (
    get_mnist_dataloader,
    get_fmnist_dataloader,
    construct_mlp,
)
from examples.mnist_uncertainty.ood_utils import (
    get_ood_input_processor,
)

parser = argparse.ArgumentParser("OOD Self-influce Score Analysis")
parser.add_argument(
    "--id-data",
    type=str,
    default="mnist",
    help="mnist or fmnist; OOD is set to the other one",
)
parser.add_argument("--damping", type=float, default=1e-5)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ood_data = "fmnist" if args.id_data == "mnist" else "mnist"
model = construct_mlp().to(DEVICE)

# Get a single checkpoint (first model_id and last epoch).
model.load_state_dict(
    torch.load(f"checkpoints/{args.id_data}_0_epoch_9.pt", map_location="cpu")
)
model.eval()

id_dataloader_fn = (
    get_mnist_dataloader if args.id_data == "mnist" else get_fmnist_dataloader
)
ood_dataloader_fn = (
    get_mnist_dataloader if ood_data == "mnist" else get_fmnist_dataloader
)
id_train_loader = id_dataloader_fn(
    batch_size=512, split="train", shuffle=False, subsample=True
)
id_query_loader = id_dataloader_fn(
    batch_size=512,
    split="valid",
    shuffle=False,
    subsample=True,
)
ood_query_loader = ood_dataloader_fn(
    batch_size=512,
    split="valid",
    shuffle=False,
    subsample=True,
)
ood_input_processor = get_ood_input_processor(
    source_data=ood_data, target_model=args.id_data
)

# Set-up
logix = LogIX(project="test")

# Gradient & Hessian logging
logix.watch(model, lora=False)
id_gen = DataIDGenerator()
for inputs, targets in id_train_loader:
    data_id = id_gen(inputs)
    with logix(data_id=data_id, log=["grad"], hessian=True, save=True):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        model.zero_grad()
        outs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
        loss.backward()
logix.finalize()

# Influence Analysis
logix.add_analysis({"influence": InfluenceFunction})

print("Computing OOD self-influence scores...")
ood_self_influence_scores = []
for ood_test_input, ood_test_target in tqdm.tqdm(ood_query_loader):
    with logix(log=["grad"]) as al:
        ood_test_input = ood_input_processor(ood_test_input)
        ood_test_input, ood_test_target = (
            ood_test_input.to(DEVICE),
            ood_test_target.to(DEVICE),
        )
        model.zero_grad()
        ood_test_out = model(ood_test_input)
        ood_test_loss = torch.nn.functional.cross_entropy(
            ood_test_out, ood_test_target, reduction="sum"
        )
        ood_test_loss.backward()
        ood_test_log = al.get_log()
    start = time.time()
    if_scores = logix.influence.compute_self_influence(
        ood_test_log, damping=args.damping
    )
    ood_self_influence_scores.append(if_scores.numpy().flatten())

print("Computing ID self-influence scores...")
id_self_influence_scores = []
for id_test_input, id_test_target in tqdm.tqdm(id_query_loader):
    with logix(log=["grad"]) as al:
        id_test_input, id_test_target = id_test_input.to(DEVICE), id_test_target.to(
            DEVICE
        )
        model.zero_grad()
        id_test_out = model(id_test_input)
        id_test_loss = torch.nn.functional.cross_entropy(
            id_test_out, id_test_target, reduction="sum"
        )
        id_test_loss.backward()
        id_test_log = al.get_log()
    start = time.time()
    if_scores = logix.influence.compute_self_influence(
        id_test_log, damping=args.damping
    )
    id_self_influence_scores.append(if_scores.numpy().flatten())

# Save
ood_sif_scores = np.concatenate(ood_self_influence_scores)
id_sif_scores = np.concatenate(id_self_influence_scores)
torch.save(ood_sif_scores, "ood_sif_scores.pt")
torch.save(id_sif_scores, "id_sif_scores.pt")
print(
    f"OOD self-influence scores: mean={ood_sif_scores.mean():.2f}, std={ood_sif_scores.std():.2f}"
)
print(
    f"ID self-influence scores: mean={id_sif_scores.mean():.2f}, std={id_sif_scores.std():.2f}"
)
