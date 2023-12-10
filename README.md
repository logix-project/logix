<p align="center">
  <a href="https://https://github.com/sangkeun00/analog/">
    <img src="assets/logo.png" alt="" width="40%" align="top">
  </a>
</p>

<div align="center">

  [![Build](https://badgen.net/badge/build/check-status/green)](#build-pipeline-status)
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/leopard-ai/betty/blob/main/LICENSE)
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="code style: black"></a>
  
</div>

```bash
git clone https://github.com/sangkeun00/analog.git; cd analog; pip install . # Install
```

## Usage
AnaLog is designed with the belief that diverse logs generated by neural networks, such as
gradient and activation logs, can be utilized for analyzing and debugging data, algorithms,
and other aspects. To use AnaLog, users simply adhere to a two-stage workflow:

1. **Logging**: Extract and save various logs to a disk
2. **Analysis**: Load logs from a disk and perform a custom analysis (e.g. influence function).

### Logging
Logging with AnaLog is as simple as adding one `with` statement to the existing
training code. AnaLog automatically extracts user-specified logs using PyTorch hooks, and
save it to a disk using a memory-mapped file.

```python
from analog import AnaLog

analog = AnaLog(project="my_project") # initialze AnaLog
analog.update({"log": ["grad"], "hessian": True, "save": True}) # set logging config
analog.watch(model) # add your model to log

for input, target in data_loader:
    with analog(data_id=input): # set data_id for the log from the current batch
        out = model(input)
        loss = loss_fn(out, target, reduction="sum")
        loss.backward()
        model.zero_grad()
analog.finalize() # finalize logging
```

### Analysis
Once logging is completed, users can simply load them from a disk, and perform any
analysis users want. We currently implemented influence function which can be used
for both training data attribution and uncertainty quantification for AI safety.

```python
from analog.analysis import InfluenceFunction

analog.eval() # enter the analysis mode
log_loader = analog.build_log_dataloader() # return PyTorch DataLoader for log data

with analog(data_id=test_input):
    test_out = model(test_input)
    test_loss = loss_fn(test_out, test_target, reduction="sum")
    test_loss.backward()
test_log = analog.get_log() # extract a log for test data

analog.add_analysis({"influence": InfluenceFunction}) # add your custom analysis

analog.influence.compute_influence_all(test_log, log_loader) # data attribution
analog.influence.compute_self_influence(test_log) # uncertainty
```

Please check out [Examples](/examples) for more advanced features!

## Features
Logs from neural networks are difficult to handle due to its huge size. For example,
gradient of *each* training data is about the same as the whole model. Therefore,
we provide various systems support to efficiently scale neural network analysis to
billion-scale models. Below are a few features that AnaLog currently support: 

- **Gradient compression** (compression ratio: 1,000-100,000x)
- **Memory-mapped-based data IO**
- **CPU offloading of logs**

## Contributing

We welcome contributions from the community. Please see our [contributing
guidelines](CONTRIBUTING.md) for details on how to contribute to AnaLog.

## Citation
To cite this repository:

```
@software{analog2024github,
  author = {Sang Keun Choe, Hwijeen Ahn, Juhan Bae, Minsoo Kang, Youngseong Chung, Kewen Zhao},
  title = {{AnaLog}: Scalable Logging and Analysis Tool for Neural Networks},
  url = {http://github.com/sangkeun00/analog},
  version = {0.0.1},
  year = {2024},
}
```

## License
AnaLog is licensed under the [Apache 2.0 License](LICENSE).
