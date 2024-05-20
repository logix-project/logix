# Data Valuation with (Large) Language Models

This directory contains the codes for running data valuation with large scale
language models like Llama3. In essence, the code ranks the pretraining data
based on the importance of each data point in the generation of a target sentence.
The procedure to rank the data points is as follows:

### Data Preparation
We here provide a simple code to generate the response of Llama3-8B-Instruct
given the user prompt. Responses, along with prompts, will be saved in
`custom_data` in a json format and will be used for data valuation later.

```python
python generate.py
```

### Extract Log

`extract_log.py` extracts the Hessian, projected gradients for each (pre)training
data point using LoGra, and saves them in memory-mapped files. Users can plug in
their own datasets just by rewriting the data loading part.

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
