import torch

from train import (
    get_mnist_dataloader,
    get_fmnist_dataloader,
    construct_mlp,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DAMPING = 0.01
LISSA_ITER = 100000


def single_checkpoint_influence(data_name="mnist", eval_idxs=(0,)):
    model = construct_mlp().to(DEVICE)

    # Get a single checkpoint (first model_id and last epoch).
    model.load_state_dict(
        torch.load(f"checkpoints/{data_name}_0_epoch_9.pt", map_location="cpu")
    )
    model.eval()

    if data_name == "mnist":
        train_loader = get_mnist_dataloader(
            batch_size=512, split="train", shuffle=False, subsample=True
        )
        sb_train_loader = get_mnist_dataloader(
            batch_size=1, split="train", shuffle=False, subsample=True
        )
        lissa_train_loader = get_mnist_dataloader(
            batch_size=8, split="train", shuffle=True, subsample=True
        )
        query_loader = get_mnist_dataloader(
            batch_size=1, split="valid", shuffle=False, indices=eval_idxs
        )
    else:
        train_loader = get_fmnist_dataloader(
            batch_size=512, split="train", shuffle=False, subsample=True
        )
        sb_train_loader = get_fmnist_dataloader(
            batch_size=1, split="train", shuffle=False, subsample=True
        )
        lissa_train_loader = get_fmnist_dataloader(
            batch_size=8, split="train", shuffle=True, subsample=True
        )
        query_loader = get_fmnist_dataloader(
            batch_size=1, split="valid", shuffle=False, indices=eval_idxs
        )

    from analog import AnaLog

    analog = AnaLog(project="test")
    analog.watch(model, type_filter=[torch.nn.Linear])

    for inputs, targets in train_loader:
        with analog(data_id=inputs, log=["forward", "backward"], save=False):
            model.zero_grad()
            outs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
            loss.backward()
    analog.finalize()
    hs = analog.get_hessian_state()
    print(hs)


if __name__ == "__main__":
    single_checkpoint_influence(data_name="mnist")
