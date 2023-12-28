import argparse
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from analog import AnaLog
from analog.analysis import InfluenceFunction
from analog.utils import DataIDGenerator
from tqdm import tqdm

from utils import construct_model, get_loaders, set_seed

parser = argparse.ArgumentParser("GPT2 Influence Analysis")
parser.add_argument("--data_name", type=str, default="wiki")
parser.add_argument("--eval-idxs", type=int, nargs="+", default=[0])
parser.add_argument("--damping", type=float, default=1e-5)
args = parser.parse_args()

set_seed(0)
accelerator = Accelerator()

# model
model = construct_model()
model.load_state_dict(
    torch.load(f"files/checkpoints/0/{args.data_name}_epoch_3.pt", map_location="cpu")
)
model.eval()

# data
# _, eval_train_loader, test_loader = get_loaders()
_, eval_train_loader, test_loader = get_loaders(
    eval_batch_size=8,
    # train_indices=list(range(32)),
    valid_indices=list(range(32)),
)

mode, eval_train_loader = accelerator.prepare(model, eval_train_loader)

# Set-up
analog = AnaLog(project="test", config="config.yaml")

# Hessian logging
# LM head in GPT2 is nn.Linear so we need to filter it out
modules_to_watch = []
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        if "lm_head" in n:
            continue
        modules_to_watch.append(n)

analog.watch(model, name_filter=modules_to_watch)
analog.setup({"statistic": "kfac"})
id_gen = DataIDGenerator(mode="index")
for epoch in range(2):
    for batch in tqdm(eval_train_loader, desc="Hessian logging"):
        data_id = id_gen(batch["input_ids"])
        inputs = (
            batch["input_ids"],
            batch["attention_mask"],
        )
        targets = batch["labels"]
        with analog(data_id=data_id, mask=inputs[-1]):
            model.zero_grad()
            lm_logits = model(*inputs)

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
                ignore_index=-100,
            )
            loss.backward()
    analog.finalize()
    if epoch == 0:
        analog.add_lora(model)
        analog.update({"log": "grad", "save": "grad", "statistic": "kfac"})

# Compute influence
analog.eval()
log_loader = analog.build_log_dataloader()
analog.add_analysis({"influence": InfluenceFunction})
test_iter = iter(test_loader)
test_batch = next(test_iter)
with analog(data_id=id_gen(test_batch["input_ids"]), mask=test_batch["attention_mask"]):
    test_inputs = (
        test_batch["input_ids"],
        test_batch["attention_mask"],
    )
    test_targets = test_batch["labels"]
    model.zero_grad()
    test_logits = model(*test_inputs)

    test_shift_logits = test_logits[..., :-1, :].contiguous()
    test_shift_labels = test_targets[..., 1:].contiguous()
    test_loss = F.cross_entropy(
        test_shift_logits.view(-1, shift_logits.size(-1)),
        test_shift_labels.view(-1),
        reduction="sum",
        ignore_index=-100,
    )
    test_loss.backward()

test_log = analog.get_log()

start = time.time()
if_scores = analog.influence.compute_influence_all(test_log, log_loader)
print("Computation time:", time.time() - start)

# Save
log_dir = analog.config.get_storage_config()["log_dir"]
save_path = f"{log_dir}/if_analog.pt"
torch.save(if_scores, save_path)
