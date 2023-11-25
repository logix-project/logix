import time
import argparse

import torch
from analog import AnaLog
from analog.utils import DataIDGenerator
from analog.analysis import InfluenceFunction

from train import (
    get_mnist_dataloader,
    get_fmnist_dataloader,
    construct_mlp,
)

parser = argparse.ArgumentParser("MNIST Influence Analysis")
parser.add_argument("--data_name", type=str, default="mnist", help="mnist or fmnist")
parser.add_argument("--eval-idxs", type=int, nargs="+", default=[0])
parser.add_argument("--damping", type=float, default=1e-5)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = construct_mlp(data_name=args.data_name).to(DEVICE)
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

analog = AnaLog(project="test")

analog.watch(model)
analog_kwargs = {"log": ["grad"], "hessian": True, "save": False}
id_gen = DataIDGenerator()
# Epoch 0: Compute Hessian
# Epoch 1: Gradient logging w/ PCA+LoRA
for epoch in range(2):
    for inputs, targets in train_loader:
        data_id = id_gen(inputs)
        with analog(data_id=data_id, **analog_kwargs):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            model.zero_grad()
            outs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
            loss.backward()
    analog.finalize()
    if epoch == 0:
        analog_kwargs.update({"save": True})
        analog.add_lora(model, parameter_sharing=False)

log_loader = analog.build_log_dataloader()

analog.add_analysis({"influence": InfluenceFunction})
query_iter = iter(query_loader)
with analog(log=["grad"], test=True) as al:
    test_input, test_target = next(query_iter)
    test_input, test_target = test_input.to(DEVICE), test_target.to(DEVICE)
    model.zero_grad()
    test_out = model(test_input)
    test_loss = torch.nn.functional.cross_entropy(
        test_out, test_target, reduction="sum"
    )
    test_loss.backward()
    test_log = al.get_log()
start = time.time()
if_scores = analog.influence.compute_influence_all(
    test_log, log_loader, damping=args.damping
)
_, top_influential_data = torch.topk(if_scores, k=10)

# Save
if_scores = if_scores.numpy().tolist()
torch.save(if_scores, "if_analog_lora64_pca.pt")
print("Computation time:", time.time() - start)
print("Top influential data indices:", top_influential_data.numpy().tolist())
