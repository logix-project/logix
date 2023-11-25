import time

import torch
import torch.nn.functional as F
from analog import AnaLog
from analog.analysis import InfluenceFunction
from analog.utils import DataIDGenerator
from tqdm import tqdm

from pipeline import construct_model, get_loaders
from utils import set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0)


def single_checkpoint_influence(
    data_name: str,
    model_name: str,
    ckpt_path: str,
    save_name: str,
    train_batch_size=4,
    test_batch_size=4,
    train_indices=None,
    test_indices=None,
):
    # model
    model = construct_model(model_name, ckpt_path)
    model.to(DEVICE)
    model.eval()

    # data
    _, eval_train_loader, test_loader = get_loaders(data_name=data_name)

    # Set-up
    analog = AnaLog(project="test", config="/data/tir/projects/tir6/general/hahn2/analog/examples/bert_influence/config.yaml")

    # Hessian logging
    analog.watch(model, type_filter=[torch.nn.Linear], lora=False)
    id_gen = DataIDGenerator()
    for batch in tqdm(eval_train_loader, desc="Hessian logging"):
        data_id = id_gen(batch["input_ids"])
        with analog(data_id=data_id, log=[], save=False):
            inputs = (
                batch["input_ids"].to(DEVICE),
                batch["token_type_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
            )
            model.zero_grad()
            outputs = model(*inputs)

            logits = outputs.view(-1, outputs.shape[-1])
            labels = batch["labels"].view(-1).to(DEVICE)
            loss = F.cross_entropy(
                logits, labels, reduction="sum", ignore_index=-100
            )
            loss.backward()
    analog.finalize()

    # Compressed gradient logging
    analog.add_lora(model, parameter_sharing=False)
    for batch in tqdm(eval_train_loader, desc="Compressed gradient logging"):
        data_id = id_gen(batch["input_ids"])
        with analog(data_id=data_id, log=["grad"], save=True):
            inputs = (
                batch["input_ids"].to(DEVICE),
                batch["token_type_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
            )
            model.zero_grad()
            outputs = model(*inputs)

            logits = outputs.view(-1, outputs.shape[-1])
            labels = batch["labels"].view(-1).to(DEVICE)
            loss = F.cross_entropy(
                logits, labels, reduction="sum", ignore_index=-100,
            )
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

        test_logits = test_outputs.view(-1, outputs.shape[-1])
        test_labels = test_batch["labels"].view(-1).to(DEVICE)
        test_loss = F.cross_entropy(
            test_logits, test_labels, reduction="sum", ignore_index=-100,
        )
        test_loss.backward()

        test_log = al.get_log()

    start = time.time()
    if_scores = analog.influence.compute_influence_all(test_log, log_loader)
    print("Computation time:", time.time() - start)

    # Save
    torch.save(if_scores, "if_analog.pt")


def main():
    data_name = "sst2"
    model_name = "bert-base-uncased"
    ckpt_path = "/data/tir/projects/tir6/general/hahn2/analog/examples/bert_influence/files/checkpoints/0/sst2_epoch_3.pt"
    save_name = "sst2_score_if.pt"

    single_checkpoint_influence(
        data_name=data_name,
        model_name=model_name,
        ckpt_path=ckpt_path,
        save_name=save_name,
    )
