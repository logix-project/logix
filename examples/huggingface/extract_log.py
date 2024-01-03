import argparse

from tqdm import tqdm
import analog
from analog import AnaLogTrainer
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import default_data_collator

from utils import construct_model, get_datasets, set_seed



def main():
    parser = argparse.ArgumentParser("GLUE Influence Analysis")
    parser.add_argument("--project", type=str, default="sst2")
    parser.add_argument("--config_path", type=str, default="./config.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_name", type=str, default="sst2")
    args = parser.parse_args()

    set_seed(0)

    # prepare model & data loader
    model, tokenizer = construct_model(
        args.data_name, ckpt_path=f"files/checkpoints/0/{args.data_name}_epoch_3.pt"
    )
    model.eval()
    train_dataset = get_datasets(args.data_name, train_indices=list(range(256)))[1]

    # AnaLog
    run = analog.init(args.project, config=args.config_path)
    scheduler = analog.AnaLogScheduler(run, lora=True)
    analog.watch(model)

    trainer = AnaLogTrainer(
        run,
        scheduler,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    # for _ in scheduler:
    #     for batch in tqdm(train_loader, desc="Hessian logging"):
    #         data_id = tokenizer.batch_decode(batch["input_ids"])
    #         labels = batch.pop("labels").view(-1)
    #         _ = batch.pop("idx")
    #         with run(data_id=data_id, mask=batch["attention_mask"]):
    #             model.zero_grad()
    #             outputs = model(**batch)
    #             logits = outputs.view(-1, outputs.shape[-1])
    #             loss = F.cross_entropy(
    #                 logits, labels, reduction="sum", ignore_index=-100
    #             )
    #             accelerator.backward(loss)
    #     analog.finalize()


if __name__ == "__main__":
    main()
