# We use an example from the TRAK repository:
# https://github.com/MadryLab/trak/blob/main/examples/cifar_quickstart.ipynb.


import os

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from tqdm.auto import tqdm
from utils import construct_rn9, get_cifar10_dataloader, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    loader,
    lr=0.4,
    epochs=24,
    momentum=0.9,
    weight_decay=5e-4,
    lr_peak_epoch=5,
    save_name="default",
    model_id=0,
    save=True,
):
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loader)
    # Cyclic LR with single triangle
    lr_schedule = np.interp(
        np.arange((epochs + 1) * iters_per_epoch),
        [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
        [0, 1, 0],
    )
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss()

    for epoch in tqdm(range(epochs)):
        set_seed(model_id * 10_061 + epoch + 1)
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(images)
                loss = loss_fn(out, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

    if save:
        torch.save(
            model.state_dict(),
            f"checkpoints/{save_name}_{model_id}_epoch_{epochs-1}.pt",
        )

    return model


def main(dataset="cifar10"):
    os.makedirs("checkpoints", exist_ok=True)

    if dataset == "cifar10":
        train_loader = get_cifar10_dataloader(
            batch_size=512, split="train", shuffle=True, subsample=True
        )
        valid_loader = get_cifar10_dataloader(
            batch_size=512, split="val", shuffle=False
        )
    else:
        raise NotImplementedError

    # you can modify the for loop below to train more models
    for model_id in tqdm(range(1), desc="Training models.."):
        model = construct_rn9().to(memory_format=torch.channels_last).to(DEVICE)

        model = train(model, train_loader, save_name=dataset, model_id=model_id)

        model = model.eval()

        model.eval()
        with torch.no_grad():
            total_correct, total_num = 0.0, 0.0
            for images, labels in tqdm(valid_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                with autocast():
                    out = model(images)
                    total_correct += out.argmax(1).eq(labels).sum().cpu().item()
                    total_num += images.shape[0]

            print(f"Accuracy: {total_correct / total_num * 100:.1f}%")


if __name__ == "__main__":
    main()
