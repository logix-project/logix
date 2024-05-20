# Influence Analysis with BERT

Influence analysis with BERT for datasets in GLUE benchmark can be
done in three steps:

1. Train your model
2. Extract training logs (i.e. gradient for all training data)
3. Compute influence scores

For each step, we provide a python script below:

### Train

```bash
python train.py
```

### Log Extraction

```bash
python extract_log.py
```

### Influence Function Analysis

```bash
python compute_influence.py
```
