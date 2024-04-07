<p align="center">
  <a href="https://github.com/sangkeun00/logix/">
    <img src="assets/logo_light.png" alt="" width="40%" align="top" style="border-radius: 10px; padding-left: 120px; padding-right: 120px; background-color: white;">
  </a>
</p>

<p align="center">
  <em><strong>LogiX</strong>: Logging for Interpretable and Explainable AI <br></em>
</p>

<div align="center">

  [![Build](https://badgen.net/badge/build/check-status/green)](#build-pipeline-status)
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/leopard-ai/betty/blob/main/LICENSE)
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="code style: black"></a>
  <a href="https://discord.gg/3vTgFnFX"><img alt="Discord" src="https://img.shields.io/discord/1159141589738868796?logo=discord&label=Discord&color=white&link=https%3A%2F%2Fdiscord.gg%2F3vTgFnFX"></a>

</div>

```bash
git clone https://github.com/sangkeun00/logix.git; cd logix; pip install . # Install
```

## Usage
LogiX is designed with the belief that diverse logs generated by neural networks, such as
gradient and activation, can be utilized for analyzing and debugging data, algorithms,
and other aspects. To use LogiX, users simply adhere to a two-stage workflow:

1. **Logging**: Extract and save various logs (e.g. per-sample gradient, activation) to disk.
2. **Analysis**: Load logs from disk and perform custom analysis (e.g. influence function).

### Logging
Logging with LogiX is as simple as adding one `with` statement to the existing
training code. LogiX automatically extracts user-specified logs using PyTorch hooks, and
saves it to disk using a memory-mapped file.

```python
import logix

run = logix.init(project="my_project") # initialze LogiX
logix.setup({"log": "grad", "save": "grad", "statistic": "kfac"}) # set logging config
logix.watch(model) # add your model to log

for input, target in data_loader:
    with run(data_id=input): # set data_id for the log from the current batch
        out = model(input)
        loss = loss_fn(out, target, reduction="sum")
        loss.backward()
        model.zero_grad()
logix.finalize() # finalize logging
```

### Analysis
Once logging is completed, the user can simply load them from disk, and perform any
analysis the user may want. We have currently implemented influence function, which can be used
for both training data attribution and uncertainty quantification for AI safety.

```python
from logix.analysis import InfluenceFunction

logix.eval() # enter analysis mode
log_loader = logix.build_log_dataloader() # return PyTorch DataLoader for log data

with logix(data_id=test_input):
    test_out = model(test_input)
    test_loss = loss_fn(test_out, test_target, reduction="sum")
    test_loss.backward()
test_log = logix.get_log() # extract a log for test data

logix.influence.compute_influence_all(test_log, log_loader) # data attribution
logix.influence.compute_self_influence(test_log) # uncertainty
```

### HuggingFace Integration
Our software design allows for the seamless integration with HuggingFace's
[Transformer](https://github.com/huggingface/transformers/tree/main), a popular DL framework
that conveniently handles distributed training, data loading, etc. We plan to support more
frameworks (e.g. Lightning) in the future!

```python
from transformers import Trainer, Seq2SeqTrainer
from logix.huggingface import patch_trainer, LogiXArguments

logix_args = LogiXArguments(project, config, lora=True, ekfac=True)
LogiXTrainer = patch_trainer(Trainer)

trainer = LogiXTrainer(logix_args=logix_args, # pass LogiXArguments as TrainingArguments
                        model=model,
                        train_dataset=train_dataset,
                        *args,
                        **kwargs)

# Instead of trainer.train(),
trainer.extract_log()
trainer.influence()
trainer.self_influence()
```

Please check out [Examples](/examples) for more advanced features!


## Features
Logs from neural networks are difficult to handle due to the large size. For example,
the size of the gradient of *each* training datapoint is about as large as the whole model. Therefore,
we provide various systems support to efficiently scale neural network analysis to
billion-scale models. Below are a few features that LogiX currently supports:

- **Gradient compression** (compression ratio: 1,000-100,000x)
- **Memory-map-based data IO**
- **CPU offloading of logs**

## Compatability
| DistributedDataParallel| Mixed Precision| Gradient Checkpointing | torch.compile  | FSDP           |
|:----------------------:|:--------------:|:----------------------:|:-------------:|:--------------:|
| ✅                     | ✅             | ✅                    | ✅           |   ✅             |

## Contributing

We welcome contributions from the community. Please see our [contributing
guidelines](CONTRIBUTING.md) for details on how to contribute to LogiX.

## Citation
To cite this repository:

```
@software{logix2024github,
  author = {Sang Keun Choe, Hwijeen Ahn, Juhan Bae, Minsoo Kang, Youngseog Chung, Kewen Zhao},
  title = {{LogiX}: Scalable Logging and Analysis Tool for Neural Networks},
  url = {http://github.com/sangkeun00/logix},
  version = {0.0.1},
  year = {2024},
}
```

## License
LogiX is licensed under the [Apache 2.0 License](LICENSE).
