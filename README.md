<p align="center">
  <a href="https://github.com/logix-project/logix">
    <img src="assets/logo.jpg" alt="" width="40%" align="top" style="border-radius: 10px; padding-left: 120px; padding-right: 120px; background-color: white;">
  </a>
</p>

<p align="center">
  <em><strong>LogIX</strong>: Logging for Interpretable and Explainable AI <br></em>
</p>

<div align="center">

  [![Build](https://badgen.net/badge/build/check-status/green)](#build-pipeline-status)
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/logix-project/logix/blob/main/LICENSE)
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="code style: black"></a>
</div>

```bash
pip install logix-ai
```


> [!WARNING]
> This repository is under active development. If you have suggestions or find bugs in LogIX, please open a GitHub issue or reach out.


## Basics
It turns out that most _interpretable & explainable AI_ research (e.g., training data valuation/attribution,
saliency maps, mechanistic interpretability) simply require **(1)** intercepting various training logs
(e.g., activation, gradient) and **(2)** doing some computational analyses with these logs. Therefore,
**LogIX** focuses on simple, efficient, and interoperable logging of training artifacts for maximal
flexibility, while providing some pre-implemented interpretability algorithm (e.g., influence functions)
for general users.


## Usage
### Logging
Training log extraction with LogIX is as simple as adding one `with` statement to the existing
training code. LogIX automatically extracts user-specified logs using PyTorch hooks, and stores
it as a tuple of `([data_ids], log[module_name][log_type])`. If needed, LogIX writes these logs
to disk efficiently with memory-mapped files.

```python
import logix

# Initialze LogIX
run = logix.init(project="my_project")

# Users can specify artifacts they want to log
run.setup({"log": "grad", "save": "grad", "statistic": "kfac"})

# Users can specify specific modules they want to track logs for
run.watch(model, name_filter=["mlp"], type_filter=[nn.Linear])

for input, target in data_loader:
    # Set data_id for the log from the current batch
    with run(data_id=input):
        out = model(input)
        loss = loss_fn(out, target, reduction="sum")
        loss.backward()
        model.zero_grad()

    # Access log extracted in the LogIX context block
    log = run.get_log() # (data_id, log_dict)
    # For example, users can print gradient for the specific module
    # print(log[1]["model.layers.23.mlp.down_proj"]["grad"])
    # or perform any custom analysis

# Synchronize statistics (e.g. grad covariance) and
# write remaining logs to disk
run.finalize()
```

### Training Data Attribution
As a part of our initial research, we implemented influence functions using LogIX. We plan to provide more
pre-implemented interpretability algorithms if there is a demand.

```python
# Build PyTorch DataLoader from saved log data
log_loader = run.build_log_dataloader()

with run(data_id=test_input):
    test_out = model(test_input)
    test_loss = loss_fn(test_out, test_target, reduction="sum")
    test_loss.backward()
# Extract a log for test data
test_log = run.get_log()

run.influence.compute_influence_all(test_log, log_loader) # Data attribution
run.influence.compute_self_influence(test_log) # Uncertainty estimation
```

### HuggingFace Integration
Our software design allows for the seamless integration with HuggingFace's
[Transformer](https://github.com/huggingface/transformers/tree/main), a popular DL framework
that conveniently handles distributed training, data loading, etc. We plan to support more
frameworks (e.g. Lightning) in the future.

```python
from transformers import Trainer, Seq2SeqTrainer
from logix.huggingface import patch_trainer, LogIXArguments

logix_args = LogIXArguments(project, config, lora=True, hessian="raw", save="grad")
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
@software{logix2024github,
  author = {Sang Keun Choe, Hwijeen Ahn, Juhan Bae, Minsoo Kang, Youngseog Chung, Kewen Zhao},
  title = {{LogIX}: Scalable Logging and Analysis Tool for Neural Networks},
  url = {https://github.com/logix-project/logix},
  version = {0.1.0},
  year = {2024},
}
```

## License
LogIX is licensed under the [Apache 2.0 License](LICENSE).
