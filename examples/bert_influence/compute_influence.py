import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from analog import AnaLog, AnaLogScheduler
from analog.analysis import InfluenceFunction
from analog.utils import DataIDGenerator

from utils import construct_model, get_loaders

parser = argparse.ArgumentParser("GLUE Influence Analysis")
parser.add_argument("--data_name", type=str, default="sst2", options=["sst2", "qnli"])
parser.add_argument(
    "--num_train_data",
    type=int,
    default=None,
    help="If set to not None, only inspect the first num_train_data samples.",
)
parser.add_argument(
    "--num_test_data",
    type=int,
    default=None,
    help="If set to not None, only inspect the first num_test_data samples.",
)
parser.add_argument(
    "--damping", type=float, default=None, help="Damping factor for influence function."
)
parser.add_argument("--ekfac", action="store_true")
parser.add_argument("--lora", action="store_true")
parser.add_argument(
    "--sample", action="store_true", help="Toggle this on to use true fisher."
)
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--project_name", type=str, default=None)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = construct_model(args.data_name)
model_path = f"files/checkpoints/0/{args.data_name}_epoch_3.pt"
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.to(DEVICE)
model.eval()

# data
_, eval_train_loader, test_loader = get_loaders(
    data_name=args.data_name,
    train_indices=args.num_train_data and list(range(args.num_train_data)),
    valid_indices=args.num_test_data and list(range(args.num_test_data)),
)

# Set-up
analog = AnaLog(project=args.project_name, config=args.config)
al_scheduler = AnaLogScheduler(
    analog, ekfac=args.ekfac, lora=args.lora, sample=args.sample
)

# Step 1. Collect logs
analog.watch(model)
id_gen = DataIDGenerator(mode="index")
for epoch in al_scheduler:
    sample = True if epoch < (len(al_scheduler) - 1) and args.sample else False
    for batch in tqdm(
        eval_train_loader,
        desc="Collecting statistics (covariances) and saving logs (compressed gradients)",
    ):
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

            if sample:
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(outputs, dim=-1)
                    labels = torch.multinomial(probs, num_samples=1).flatten()
            else:
                labels = batch["labels"].view(-1).to(DEVICE)

            loss = F.cross_entropy(
                logits, labels, reduction="sum", ignore_index=-100
            )  # influence function requires sum reduction
            loss.backward()
    analog.finalize()

# Step 2. Perform analysis
analog.eval()
log_loader = analog.build_log_dataloader()
analog.add_analysis({"influence": InfluenceFunction})
if_scores_list = []
for test_batch in tqdm(test_loader, desc="Computing Influence"):
    with analog(log=["grad"], test=True) as al:
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
        if_scores = analog.influence.compute_influence_all(
            test_log, log_loader, damping=args.damping
        )
        if_scores_list.append(if_scores)
if_scores = torch.cat(if_scores_list, dim=0)
save_path = "if_scores.pt"
torch.save(if_scores, save_path)
print(f"Saved influence scores of size {if_scores.size()} to {save_path}")
