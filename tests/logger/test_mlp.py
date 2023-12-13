import unittest
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig

from analog import AnaLog


def create_mlp(input_size, hidden_size, num_classes):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, num_classes),
    )
    return model


class TestMLPGradients(unittest.TestCase):
    def setUp(self):
        input_size = 4
        hidden_size = 8
        num_classes = 5

        self.model = create_mlp(input_size, hidden_size, num_classes)
        self.func_model = create_mlp(input_size, hidden_size, num_classes)
        self.func_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.func_params = dict(self.func_model.named_parameters())
        self.func_buffers = dict(self.func_model.named_buffers())

        self.model.eval()
        self.func_model.eval()

    def test_per_sample_gradient(self):
        # Instantiate AnaLog
        analog = AnaLog(project="test")
        analog.watch(self.model)

        # Input and target for batch size of 4
        inputs = torch.randn(4, 4)  # Dummy input
        labels = torch.tensor([1, 3, 0, 2])  # Dummy labels
        batch = (inputs, labels)

        # functorch per-sample gradient
        def compute_loss_func(_params, _buffers, _batch):
            _output = torch.func.functional_call(
                self.func_model,
                (_params, _buffers),
                args=(_batch[0],),
            )
            _loss = F.cross_entropy(_output, _batch[1])
            return _loss

        func_compute_grad = torch.func.grad(compute_loss_func, has_aux=False)

        grads_dict = torch.func.vmap(
            func_compute_grad,
            in_dims=(None, None, 0),
            randomness="same",
        )(self.func_params, self.func_buffers, batch)

        # Forward pass with original model
        analog.update({"log": ["grad"], "hessian": False, "save": False})
        with analog(data_id=inputs):
            self.model.zero_grad()
            output = self.model(inputs)
            loss = F.cross_entropy(output, labels, reduction="sum")
            loss.backward()
        analog_grads_dict = analog.get_log()

        for module_name in analog_grads_dict:
            analog_grad = analog_grads_dict[module_name]
            func_grad = grads_dict[module_name + ".weight"]
            self.assertTrue(torch.allclose(analog_grad, func_grad, atol=1e-6))

    def test_per_sample_gradient_with_gradient_checkpoint(self):
        from torch.utils.checkpoint import checkpoint_sequential

        # Instantiate AnaLog
        analog = AnaLog(project="test")
        analog.watch(self.model)

        # Input and target for batch size of 4
        inputs = torch.randn(4, 4)  # Dummy input
        labels = torch.tensor([1, 3, 0, 2])  # Dummy labels
        batch = (inputs, labels)

        # functorch per-sample gradient
        def compute_loss_func(_params, _buffers, _batch):
            _output = torch.func.functional_call(
                self.func_model,
                (_params, _buffers),
                args=(_batch[0],),
            )
            _loss = F.cross_entropy(_output, _batch[1])
            return _loss

        func_compute_grad = torch.func.grad(compute_loss_func, has_aux=False)

        grads_dict = torch.func.vmap(
            func_compute_grad,
            in_dims=(None, None, 0),
            randomness="same",
        )(self.func_params, self.func_buffers, batch)

        # Forward pass with original model
        analog.update({"log": ["grad"], "hessian": False, "save": False})
        with analog(data_id=inputs):
            self.model.zero_grad()
            output = checkpoint_sequential(
                self.model, segments=2, input=inputs, use_reentrant=False
            )
            loss = F.cross_entropy(output, labels, reduction="sum")
            loss.backward()
        analog_grads_dict = analog.get_log()

        for module_name in analog_grads_dict:
            analog_grad = analog_grads_dict[module_name]
            func_grad = grads_dict[module_name + ".weight"]
            self.assertTrue(torch.allclose(analog_grad, func_grad, atol=1e-6))

    def test_per_sample_gradient_with_compile(self):
        # Instantiate AnaLog
        analog = AnaLog(project="test")
        analog.watch(self.model)

        compiled_model = torch.compile(self.model)

        # Input and target for batch size of 4
        inputs = torch.randn(4, 4)  # Dummy input
        labels = torch.tensor([1, 3, 0, 2])  # Dummy labels
        batch = (inputs, labels)

        # functorch per-sample gradient
        def compute_loss_func(_params, _buffers, _batch):
            _output = torch.func.functional_call(
                self.func_model,
                (_params, _buffers),
                args=(_batch[0],),
            )
            _loss = F.cross_entropy(_output, _batch[1])
            return _loss

        func_compute_grad = torch.func.grad(compute_loss_func, has_aux=False)

        grads_dict = torch.func.vmap(
            func_compute_grad,
            in_dims=(None, None, 0),
            randomness="same",
        )(self.func_params, self.func_buffers, batch)

        # Forward pass with original model
        analog.update({"log": ["grad"], "hessian": False, "save": False})
        with analog(data_id=inputs):
            compiled_model.zero_grad()
            output = compiled_model(inputs)
            loss = F.cross_entropy(output, labels, reduction="sum")
            loss.backward()
        analog_grads_dict = analog.get_log()

        for module_name in analog_grads_dict:
            analog_grad = analog_grads_dict[module_name]
            func_grad = grads_dict[module_name + ".weight"]
            self.assertTrue(torch.allclose(analog_grad, func_grad, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
