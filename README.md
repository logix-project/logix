# AnaLog
### How to use?
1. **Log** gradient-related statistics (e.g. gradient, activation)
2. **Analyze** log

### Logging
```python
from analog import AnaLog

analog = AnaLog(project="analog")
analog.watch(model)
for input, target in data_loader:
    with analog(data=input, track="activation", covariance="kfac", save=True):
        out = model(input)
        loss = loss_fn(out, target)
        loss.backward()
analog.finalize()
```

### Analysis
```python
# Debug (test_input, test_target)
with analog(track="activation", test=True) as al:
    test_out = model(test_input)
    test_loss = loss_fn(test_out, test_target)
    test_loss.backward()
    test_activations = al.get_log()
analog.if_all(test_activations)
```
