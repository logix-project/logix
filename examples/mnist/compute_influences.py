import time
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
    from analog.utils import DataIDGenerator

    analog = AnaLog(project="test", config="./examples/mnist/config.yaml")
    analog.watch(model, type_filter=[torch.nn.Linear], lora = False)
    id_gen = DataIDGenerator()

    for inputs, targets in train_loader:
        data_id = id_gen(inputs)
        with analog(data_id=data_id, log=["grad"], save=True):
            # with analog(data_id=data_id, log=["forward", "backward"], save=True):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            model.zero_grad()
            outs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
            loss.backward()
    analog.finalize()
    hs = analog.get_hessian_state()
    analog.hessian_inverse(override=False)
    print(hs)

    log_loader = analog.build_log_dataloader()

    from analog.analysis import InfluenceFunction

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
    if_scores = analog.influence.compute_influence_all(test_log, log_loader)
    if_scores = if_scores.numpy().tolist()
    torch.save(if_scores, "if_analog.pt")
    print("Computation time:", time.time() - start)
    print(sorted(if_scores)[:10], sorted(if_scores)[-10:])


if __name__ == "__main__":
    single_checkpoint_influence(data_name="mnist")
