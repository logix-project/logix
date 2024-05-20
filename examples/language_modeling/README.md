# Data Valuation with (Large) Language Models

This directory contains the codes for running data valuation with large scale
language models like LLAMA3. In essence, the code ranks the pretraining data
based on the importance of each data point in the generation of a target sentence.
The procedure to rank the data points is as follows:

### Data Preparation
Generate the model outputs that we will analyze. This is a simple code that
generates the output based on the prompt. We experimented with
`Meta-Llama-3-8B-Instruct`, `pythia-1.4b` and `gpt2-xl`. Use `generate_llama3.py`
for `Meta-Llama-3-8B-Instruct` and `generate.py` for `pythia-1.4b` and `gpt2-xl`.

```python
python generate_llama3.py
python generate.py
```

### Extract Log

`extract_log.py` extracts training gradients for each pretraining data point,
compresses them using LoGra, and saves them in files. Note that by default we
use 1B tokens from `openwebtext` data, leveraging data parallelism. An example
running command is as follows. This is the most time consuming part of the pipeline.

```python
accelerate launch --num_processes 2 --num_machines 1 --multi_gpu extract_log.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --lora random \
  --hessian raw \
  --mlp_only
  --data_name openwebtext
```
As a result, the code will generate a folder containing the compressed gradients
for each data point and other statistics necessary for running LoGra (e.g. the
random initialization of LoGra parameters, the covariance of the gradients, etc.).

### Compute Influence function
`compute_influence.py` computes the influence score for each data point, using
the compressed gradient we just generated. The specified query data (`data_name`)
is used to compute the query gradient. As we have already saved (preconditioned)
the training gradients, this is a relatively fast process.

```python
python compute_influence.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --lora random \
  --hessian raw \
  --split generated \
  --mlp_only \
  --data_name openwebtext \
  --mode cosine
```

### Analysis
Finally, we also include a minimal analysis code that extracts the top-k most
influential data points and saves them in a file. This code is `analysis.py`.

```bash
python qualitative_analysis.py
```
