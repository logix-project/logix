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
```

### Analysis
```python
# Debug (test_input, test_target)
with analog(track="activation") as a:
    test_out = model(test_input)
    test_loss = loss_fn(test_out, test_target)
    test_loss.backward()
    a.if_all()
```
