import time
import argparse

from tqdm import tqdm

import torch
import torch.nn.functional as F
from analog import AnaLog
from analog.analysis import InfluenceFunction
from analog.utils import DataIDGenerator
from tqdm import tqdm

from utils import construct_model, get_loaders, set_seed

parser = argparse.ArgumentParser("GLUE Influence Analysis")
parser.add_argument("--data_name", type=str, default="sst2")
parser.add_argument("--eval-idxs", type=int, nargs="+", default=[0])
parser.add_argument("--damping", type=float, default=1e-5)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0)

# model
model = construct_model(args.data_name)
model.load_state_dict(
    torch.load(f"files/checkpoints/0/{args.data_name}_epoch_3.pt", map_location="cpu")
)
model.to(DEVICE)
model.eval()

# data
_, eval_train_loader, test_loader = get_loaders(data_name=args.data_name)

# Set-up
analog = AnaLog(project="test", config="config.yaml")

# Hessian logging
analog.watch(model)
analog_kwargs = {"log": [], "hessian": True, "save": False}
analog.update(analog_kwargs)
id_gen = DataIDGenerator(mode="index")
for epoch in range(2):
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

            logits = outputs.view(-1, outputs.shape[-1])
            labels = batch["labels"].view(-1).to(DEVICE)
            loss = F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)
            loss.backward()
    analog.finalize()
    if epoch == 0:
        analog_kwargs.update({"save": True, "log": ["grad"]})
        analog.update(analog_kwargs)
        analog.add_lora(model)

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
torch.save(if_scores, "if_analog.pt")
