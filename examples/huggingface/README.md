# Huggingface Integration

As the software design of LogIX prioritized the compatibility with other
tools in the ML ecosystem, we were able to integrate LogIX into Huggingface
(Transformers) Trainer. This example shows how to patch HF Trainer to
enable LogIX for BERT+GLUE and GPT+WikiText examples.

### Log Extraction

```bash
python bert_log.py # BERT
python gpt_log.py  # GPT
```

### Influence Analysis

```bash
python bert_influence.py # BERT
python gpt_influence.py  # GPT
```
