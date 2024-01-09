import unittest
import torch
import torchvision
import torch.nn as nn
import numpy as np
import os

from analog import AnaLog, AnaLogScheduler
from analog.utils import DataIDGenerator
from analog.analysis import InfluenceFunction
from tests.examples.utils import get_mnist_dataloader, construct_mlp

DEVICE = torch.device("cpu")


class TestAddLora(unittest.TestCase):
    def test_add_lora(self):
        eval_idxs = (1,)
        train_idxs = [i for i in range(0, 50000, 1000)]

        model = construct_mlp().to(DEVICE)
        # Get a single checkpoint (first model_id and last epoch).
        model.load_state_dict(
            torch.load(
                f"{os.path.dirname(os.path.abspath(__file__))}/checkpoints/mnist_0_epoch_9.pt",
                map_location="cpu",
            )
        )
        model.eval()

        dataloader_fn = get_mnist_dataloader
        train_loader = dataloader_fn(
            batch_size=512, split="train", shuffle=False, indices=train_idxs
        )
        query_loader = dataloader_fn(
            batch_size=1, split="valid", shuffle=False, indices=eval_idxs
        )

        analog = AnaLog(
            project="test",
            config=f"{os.path.dirname(os.path.abspath(__file__))}/configs/lora.yaml",
        )
        # Gradient & Hessian logging
        al_scheduler = AnaLogScheduler(analog, ekfac=False, lora=False, sample=False)
        analog.watch(model)
        id_gen = DataIDGenerator()
        for epoch in al_scheduler:
            for inputs, targets in train_loader:
                data_id = id_gen(inputs)
                with analog(data_id=data_id):
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    model.zero_grad()
                    outs = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        outs, targets, reduction="sum"
                    )
                    loss.backward()
        analog.finalize()

        log_loader = analog.build_log_dataloader()

        analog.add_analysis({"influence": InfluenceFunction})
        query_iter = iter(query_loader)
        test_input, test_target = next(query_iter)
        test_id = id_gen(test_input)
        analog.eval()
        with analog(data_id=test_id):
            test_input, test_target = test_input.to(DEVICE), test_target.to(DEVICE)
            model.zero_grad()
            test_out = model(test_input)
            test_loss = torch.nn.functional.cross_entropy(
                test_out, test_target, reduction="sum"
            )
            test_loss.backward()
        test_log = analog.get_log()
        if_scores = analog.influence.compute_influence_all(
            test_log, log_loader, damping=1e-5
        )

        # Save
        if_scores = if_scores[0]
        print(if_scores)
        # torch.save(if_scores, f"{os.path.dirname(os.path.abspath(__file__))}/if_analog_lora.pt")
        if_score_saved = torch.load(
            f"{os.path.dirname(os.path.abspath(__file__))}/data/if_analog_lora.pt"
        )
        self.assertTrue(torch.allclose(if_score_saved, if_scores))


if __name__ == "__main__":
    unittest.main()
