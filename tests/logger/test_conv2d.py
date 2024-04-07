import unittest
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from logix import LogIX


class Simple2DCNN(nn.Module):
    def __init__(self, num_channels, hidden_size, num_classes):
        super(Simple2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            num_channels, hidden_size, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.fc = nn.Linear(
            hidden_size * 4 * 4, num_classes
        )  # Assuming input size is (num_channels, 14, 14)

    def forward(self, x):
        out = self.pool(self.relu(self.conv1(x)))
        out = self.relu(self.conv2(out))
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc(out)
        return out


class Test2DCNNGradients(unittest.TestCase):
    def setUp(self):
        num_channels = 1
        hidden_size = 8
        num_classes = 10
        self.model = Simple2DCNN(num_channels, hidden_size, num_classes)
        self.func_model = Simple2DCNN(num_channels, hidden_size, num_classes)
        self.func_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.func_params = dict(self.func_model.named_parameters())
        self.func_buffers = dict(self.func_model.named_buffers())

        self.model.eval()
        self.func_model.eval()

    def test_per_sample_gradient(self):
        # Instantiate LogIX
        logix = LogIX(project="test")
        logix.watch(self.model)

        # Input and target for batch size of 4
        inputs = torch.randn(
            4, 1, 8, 8
        )  # Dummy input for 2D CNN (batch_size, channels, height, width)
        labels = torch.tensor([1, 0, 1, 0])  # Dummy labels
        batch = (inputs, labels)

        # functorch per-sample gradient
        def compute_loss_func(_params, _buffers, _batch):
            _output = torch.func.functional_call(
                self.func_model,
                (_params, _buffers),
                args=(_batch[0].unsqueeze(0),),
            )
            _loss = F.cross_entropy(_output, _batch[1].unsqueeze(0))
            return _loss

        func_compute_grad = torch.func.grad(compute_loss_func, has_aux=False)

        grads_dict = torch.func.vmap(
            func_compute_grad,
            in_dims=(None, None, 0),
            randomness="same",
        )(self.func_params, self.func_buffers, batch)

        # Forward pass with original model
        logix.setup({"log": "grad"})
        with logix(data_id=inputs):
            self.model.zero_grad()
            output = self.model(inputs)
            loss = F.cross_entropy(output, labels, reduction="sum")
            loss.backward()
        _, logix_grads_dict = logix.get_log()

        for module_name in logix_grads_dict:
            logix_grad = logix_grads_dict[module_name]["grad"]
            func_grad = grads_dict[module_name + ".weight"].view(logix_grad.shape)
            self.assertTrue(torch.allclose(logix_grad, func_grad, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
