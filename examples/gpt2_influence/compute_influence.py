import argparse
import time

import torch
import torch.nn.functional as F
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0)

# model
model = construct_model()
model.load_state_dict(
    torch.load(f"files/checkpoints/0/{args.data_name}_epoch_3.pt", map_location="cpu")
)
model.to(DEVICE)
model.eval()

# data
_, eval_train_loader, test_loader = get_loaders()
import ipdb

ipdb.set_trace(context=10)

# Set-up
analog = AnaLog(project="test", config="config.yaml")

# Hessian logging
analog.watch(model)
analog_kwargs = {"log": [], "hessian": True, "save": False}
id_gen = DataIDGenerator(mode="index")
for epoch in range(2):
    for batch in tqdm(eval_train_loader, desc="Hessian logging"):
        data_id = id_gen(batch["input_ids"])
        inputs = (
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        )
        targets = batch["labels"].to(DEVICE)
        with analog(data_id=data_id, mask=inputs[-1], **analog_kwargs):
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
        analog_kwargs.update({"save": True, "log": ["grad"]})
        analog.add_lora(model, parameter_sharing=False)

# Compute influence
log_loader = analog.build_log_dataloader()
analog.add_analysis({"influence": InfluenceFunction})
test_iter = iter(test_loader)
with analog(log=["grad"], test=True) as al:
    test_batch = next(test_iter)
    test_inputs = (
        test_batch["input_ids"].to(DEVICE),
        test_batch["attention_mask"].to(DEVICE),
    )
    test_targets = test_batch["labels"].to(DEVICE)
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

    test_log = al.get_log()

start = time.time()
if_scores = analog.influence.compute_influence_all(test_log, log_loader)
print("Computation time:", time.time() - start)

# Save
torch.save(if_scores, "if_analog.pt")
