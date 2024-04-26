import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from experiments.utils import set_seed
from tests.utils import check_model_equivalence, get_num_params


@pytest.mark.parametrize(
    "data_name",
    ["sst2", "qnli", "rte"],
)
def test_glue_pipeline(data_name: str) -> None:
    from experiments.glue.pipeline import construct_model, get_loaders

    train_loader, train_eval_loader, valid_loader = get_loaders(
        data_name=data_name,
    )
    assert len(train_loader.dataset) == len(train_eval_loader.dataset)

    # Check if the dataset did not change.
    if data_name != "rte":
        assert len(train_loader.dataset) == 51_200
        assert len(train_loader.dataset) % 32 == 0
        assert len(valid_loader.dataset) == 512
        assert len(valid_loader.dataset) % 32 == 0
    else:
        assert len(train_loader.dataset) == 2432
        assert len(train_loader.dataset) % 32 == 0

    model = construct_model(data_name)
    num_params = get_num_params(model)

    # Check if the architecture did not change.
    assert num_params == 108_311_810

    # Check if the seed works correctly.
    set_seed(0)
    model1 = construct_model(data_name)
    set_seed(0)
    model2 = construct_model(data_name)
    set_seed(3)
    model3 = construct_model(data_name)
    assert check_model_equivalence(model1, model2)
    assert not check_model_equivalence(model1, model3)

    # Check if the dataset is correctly corrupted.
    _, corrupt_train_eval_loader, _ = get_loaders(data_name, do_corrupt=True)
    num_corrupt = math.ceil(0.1 * len(corrupt_train_eval_loader.dataset))
    assert len(train_eval_loader.dataset) == len(corrupt_train_eval_loader.dataset)
    assert np.all(
        np.array(train_eval_loader.dataset["labels"][:num_corrupt])
        != np.array(corrupt_train_eval_loader.dataset["labels"][:num_corrupt])
    )
    assert np.all(
        np.array(train_eval_loader.dataset["labels"][num_corrupt:])
        == np.array(corrupt_train_eval_loader.dataset["labels"][num_corrupt:])
    )

    # Check if the train batch size (and actual data) did not change.
    if data_name != "rte":
        batch = next(iter(train_loader))
        assert batch["labels"].shape[0] == 32
        batch = next(iter(train_eval_loader))
        if data_name == "sst2":
            assert torch.sum(batch["labels"]).item() == 15
        else:
            assert torch.sum(batch["labels"]).item() == 19
        batch = next(iter(corrupt_train_eval_loader))
        if data_name == "sst2":
            assert torch.sum(batch["labels"]).item() == 17
        else:
            assert torch.sum(batch["labels"]).item() == 13


def test_task() -> None:
    from experiments.glue.pipeline import construct_model
    from experiments.glue.task import (
        GlueExperimentTask,
        GlueWithAllExperimentTask,
        GlueWithLayerNormExperimentTask,
    )

    model = construct_model(data_name="sst2")
    num_params = get_num_params(model)
    task = GlueExperimentTask()
    task_with_ln = GlueWithLayerNormExperimentTask()
    task_with_all = GlueWithAllExperimentTask()

    num_influence_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert name in task.influence_modules()
            num_influence_params += get_num_params(module)
    assert num_params != num_influence_params

    num_influence_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            assert name in task_with_ln.influence_modules()
            num_influence_params += get_num_params(module)
    assert num_params != num_influence_params

    num_influence_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
            assert name in task_with_all.influence_modules()
            num_influence_params += get_num_params(module)
    assert num_params == num_influence_params
