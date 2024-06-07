<p align="center">
  <a href="https://github.com/logix-project/logix">
    <img src="assets/logo.jpg" alt="" width="40%" align="top" style="border-radius: 10px; padding-left: 120px; padding-right: 120px; background-color: white;">
  </a>
</p>

<p align="center">
  <em><strong>LogIX</strong>: Logging for Interpretable and Explainable AI <br></em>
</p>

<div align="center">

  ![PyPI - Version](https://img.shields.io/pypi/v/logix-ai)
  ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/logix-project/logix/python-test.yml)
  ![GitHub License](https://img.shields.io/github/license/logix-ai/logix)
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="code style: black"></a>
  <a href="https://arxiv.org/abs/2405.13954">![arXiv](https://img.shields.io/badge/arXiv-2405.13954-b31b1b.svg)</a>
</div>


> [!WARNING]
> This repository is under active development. If you have suggestions or find bugs in LogIX, please open a GitHub issue or reach out.


## Introduction
With a few additional lines of code, (traditional) **logging** supports tracking loss, hyperparameters, etc., providing _basic_ insights for users'
AI/ML experiments. But...can we also enable _in-depth understanding of large-scale training data_, the most important ingredient in
AI/ML, with a similar logging interface? Try out LogIX that is built upon our cutting-edge data valuation/attribution research (Support
[Huggingface Transformers](https://github.com/logix-project/logix/tree/main?tab=readme-ov-file#huggingface-integration) and
[PyTorch Lightning](https://github.com/logix-project/logix/tree/main?tab=readme-ov-file#pytorch-lightning-integration) integrations)!

- **PyPI**
```bash
pip install logix-ai
```

- **From source** (Latest, recommended)
```bash
git clone https://github.com/logix-project/logix.git
cd logix
pip install -e .
```


## Easy to Integrate

Our software design allows for the seamless integration with popular high-level frameworks including
[HuggingFace Transformer](https://github.com/huggingface/transformers/tree/main) and
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), that conveniently handles
distributed training, data loading, etc. Advanced users, who don't use high-level frameworks, can
still integrate LogIX into their existing training code similarly to any traditional logging software
(See our [Tutorial](https://github.com/logix-project/logix/tree/main?tab=readme-ov-file#getting-started)).

### 🤗 HuggingFace Integration

A full example can be found [here](https://github.com/logix-project/logix/tree/main/examples/huggingface).

```python
from transformers import Trainer, Seq2SeqTrainer
from logix.huggingface import patch_trainer, LogIXArguments

# Define LogIX arguments
logix_args = LogIXArguments(project="myproject",
                            config="config.yaml",
                            lora=True,
                            hessian="raw",
                            save="grad")

# Patch HF Trainer
LogIXTrainer = patch_trainer(Trainer)

# Pass LogIXArguments as TrainingArguments
trainer = LogIXTrainer(logix_args=logix_args,
                       model=model,
                       train_dataset=train_dataset,
                       *args,
                       **kwargs)

# Instead of trainer.train(), use
trainer.extract_log()
trainer.influence()
trainer.self_influence()
```

### ⚡ PyTorch Lightning Integration

A full example can be found [here](https://github.com/logix-project/logix/tree/main/examples/lightning).

```python
from lightning import LightningModule, Trainer
from logix.lightning import patch, LogIXArguments

class MyLitModule(LightningModule):
    ...

def data_id_extractor(batch):
    return tokenizer.batch_decode(batch["input_ids"])

# Define LogIX arguments
logix_args = LogIXArguments(project="myproject",
                            config="config.yaml",
                            lora=True,
                            hessian="raw",
                            save="grad")

# Patch Lightning Module and Trainer
LogIXModule, LogIXTrainer = patch(MyLitModule,
                                  Trainer,
                                  logix_args=logix_args,
                                  data_id_extractor=data_id_extractor)

# Use patched Module and Trainer as before
module = LogIXModule(user_args)
trainer = LogIXTrainer(user_args)

# Instead of trainer.fit(module, train_loader), use
trainer.extract_log(module, train_loader)
trainer.influence(module, train_loader)
```

## Getting Started
### Logging
Training log extraction with LogIX is as simple as adding one `with` statement to the existing
training code. LogIX automatically extracts user-specified logs using PyTorch hooks, and stores
it as a tuple of `([data_ids], log[module_name][log_type])`. If needed, LogIX writes these logs
to disk efficiently with memory-mapped files.

```python
import logix

# Initialze LogIX
run = logix.init(project="my_project")

# Specify modules to be tracked for logging
run.watch(model, name_filter=["mlp"], type_filter=[nn.Linear])

# Specify plugins to be used in logging
run.setup({"grad": ["log", "covariance"]})
run.save(True)

for batch in data_loader:
    # Set `data_id` (and optionally `mask`) for the current batch 
    with run(data_id=batch["input_ids"], mask=batch["attention_mask"]):
        model.zero_grad()
        loss = model(batch)
        loss.backward()
# Synchronize statistics (e.g. covariance) and write logs to disk
run.finalize()
```

### Training Data Attribution
As a part of our initial research, we implemented influence functions using LogIX. We plan to provide more
pre-implemented interpretability algorithms if there is a demand.

```python
# Build PyTorch DataLoader from saved log data
log_loader = run.build_log_dataloader()

with run(data_id=test_batch["input_ids"]):
    test_loss = model(test_batch)
    test_loss.backward()

test_log = run.get_log()
run.influence.compute_influence_all(test_log, log_loader) # Data attribution
run.influence.compute_self_influence(test_log) # Uncertainty estimation
```

Please check out [Examples](/examples) for more detailed examples!


## Features
Logs from neural networks are difficult to handle due to the large size. For example,
the size of the gradient of *each* training datapoint is about as large as the whole model. Therefore,
we provide various systems support to efficiently scale neural network analysis to
billion-scale models. Below are a few features that LogIX currently supports:

- **Gradient compression** (compression ratio: 1,000-100,000x)
- **Memory-map-based data IO**
- **CPU offloading of statistics**

## Compatability
| DistributedDataParallel| Mixed Precision| Gradient Checkpointing | torch.compile  | FSDP           |
|:----------------------:|:--------------:|:----------------------:|:-------------:|:--------------:|
| ✅                     | ✅             | ✅                    | ✅           |   ✅             |

## Contributing

We welcome contributions from the community. Please see our [contributing
guidelines](CONTRIBUTING.md) for details on how to contribute to LogIX.

## Citation
To cite this repository:

```
@article{choe2024your,
  title={What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions},
  author={Choe, Sang Keun and Ahn, Hwijeen and Bae, Juhan and Zhao, Kewen and Kang, Minsoo and Chung, Youngseog and Pratapa, Adithya and Neiswanger, Willie and Strubell, Emma and Mitamura, Teruko and others},
  journal={arXiv preprint arXiv:2405.13954},
  year={2024}
}
```

## License
LogIX is licensed under the [Apache 2.0 License](LICENSE).
