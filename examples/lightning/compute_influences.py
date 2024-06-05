import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from utils import get_mnist_dataloader

from logix.lightning import LogIXArguments, patch
from logix.utils import DataIDGenerator


class LITMLP(L.LightningModule):
    def __init__(self, num_inputs=784, num_classes=10):
        super().__init__()
        self.model = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, num_classes, bias=False),
        )
        self.model.load_state_dict(
            torch.load("checkpoints/mnist_0_epoch_9.pt", map_location="cpu")
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outs = self.model(inputs)
        loss = F.cross_entropy(outs, targets, reduction="sum")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


test_loader = get_mnist_dataloader(
    batch_size=1, split="valid", shuffle=False, indices=[0]
)

logix_args = LogIXArguments(
    project="lightning",
    config="config.yaml",
    lora=False,
    hessian="ekfac",
    save="grad",
    model_key="model",
    initialize_from_log=True,
    influence_damping=1e-5,
)
hash_id = DataIDGenerator()


def data_id_extractor(batch):
    return hash_id(batch[0])


LogIXModule, LogIXTrainer = patch(
    LITMLP, L.Trainer, logix_args=logix_args, data_id_extractor=data_id_extractor
)

mlp = LogIXModule()
trainer = LogIXTrainer(max_steps=1)
result = trainer.influence(mlp, test_loader)

_, top_influential_data = torch.topk(result["influence"], k=10)
if_scores = result["influence"].cpu().numpy().tolist()[0]
torch.save(if_scores, "if_logix.pt")
print("Top influential data indices:", top_influential_data.cpu().numpy().tolist())
