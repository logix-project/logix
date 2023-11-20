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

    dataloader_fn = (
        get_mnist_dataloader if data_name == "mnist" else get_fmnist_dataloader
    )
    train_loader = dataloader_fn(
        batch_size=512, split="train", shuffle=False, subsample=True
    )
    query_loader = dataloader_fn(
        batch_size=1, split="valid", shuffle=False, indices=eval_idxs
    )

    # Set-up
    from analog import AnaLog
    from analog.utils import DataIDGenerator

    analog = AnaLog(project="test", config="config.yaml")

    # Gradient & Hessian logging
    analog.watch(model, name_filter=["1", "3"])
    id_gen = DataIDGenerator()
    for inputs, targets in train_loader:
        data_id = id_gen(inputs)
        with analog(data_id=data_id, log=["grad"], save=True):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            model.zero_grad()
            outs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
            loss.backward()
    analog.finalize(hessian_inverse=True, hessian_override=True)

    # Influence Analysis
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

    # Save
    if_scores = if_scores.numpy().tolist()
    torch.save(if_scores, "if_analog.pt")
    print("Computation time:", time.time() - start)
    print(sorted(if_scores)[:10], sorted(if_scores)[-10:])


if __name__ == "__main__":
    single_checkpoint_influence(data_name="mnist")
