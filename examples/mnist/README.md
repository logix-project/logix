# MNIST

### Step 1: Train
Since influence functions are typically computed at the final model
weight, we first train the model (MLP) on the (small) MNIST dataset
before performinig influence analyses.

```bash
python train.py
```

### Step 2: Log extraction & Influence analysis
With the trained model, we first extract and save logs (e.g.
Hessian, gradeint) to disk, and use it to compute influence
scores. Users can specify the gradient projection strategy
(e.g. LoGra) and the Hessian computation strategy.

```bash
python compute_influences.py --lora none --hessian kfac --save grad
```
