# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time

import torch
from tqdm import tqdm
from train import construct_mlp, get_fmnist_dataloader, get_mnist_dataloader

from logix import LogIX, LogIXScheduler
from logix.analysis import InfluenceFunction
from logix.utils import DataIDGenerator

parser = argparse.ArgumentParser("MNIST Influence Analysis")
parser.add_argument("--data", type=str, default="mnist", help="mnist or fmnist")
parser.add_argument("--eval-idxs", type=int, nargs="+", default=[0])
parser.add_argument("--damping", type=float, default=1e-5)
parser.add_argument("--hessian", type=str, default="none")
parser.add_argument("--lora", type=str, default="none")
parser.add_argument("--save", type=str, default="grad")
parser.add_argument("--flatten", action="store_true")
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
test_loader = dataloader_fn(
    batch_size=1, split="valid", shuffle=False, indices=args.eval_idxs
)

logix = LogIX(project="test", config="./config.yaml")
scheduler = LogIXScheduler(logix, lora=args.lora, hessian=args.hessian, save=args.save)

# Gradient & Hessian logging
logix.watch(model)
id_gen = DataIDGenerator()

for epoch in scheduler:
    for inputs, targets in tqdm(train_loader):
        with logix(data_id=id_gen(inputs)):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            model.zero_grad()
            outs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
            loss.backward()
    logix.finalize()

# Influence Analysis
log_loader = logix.build_log_dataloader(
    batch_size=64, num_workers=0, flatten=args.flatten
)

# logix.add_analysis({"influence": InfluenceFunction})
logix.setup({"grad": ["log"]})
logix.eval()
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
    result = logix.influence.compute_influence_all(
        test_log, log_loader, damping=args.damping
    )
    break
_, top_influential_data = torch.topk(result["influence"], k=10)

# Save
if_scores = result["influence"].cpu().numpy().tolist()[0]
torch.save(if_scores, "if_logix.pt")
print("Top influential data indices:", top_influential_data.cpu().numpy().tolist())
