# AnaLog
### How to use?
1. **Log** gradient-related statistics (e.g. gradient, activation)
2. **Analyze** log

### Logging
```diff
  import torch
  import torch.nn.functional as F
  from datasets import load_dataset
+ from analog import AnaLog

  model = torch.nn.Transformer().to(device)
  optimizer = torch.optim.Adam(model.parameters())
  dataset = load_dataset('my_dataset')
  data = torch.utils.data.DataLoader(dataset, shuffle=True)



+ analog = AnaLog(project="analog")
+ analog.watch(model)
  for input, target in data_loader:
+     with analog(data=input, track="activation", covariance="kfac", save=True):
          out = model(input)
          loss = loss_fn(out, target)
          loss.backward()
+ analog.finalize()
```

### Analysis
```diff
  # debug (test_input, test_target)
+ with analog(track="activation", test=True) as al:
      test_out = model(test_input)
      test_loss = loss_fn(test_out, test_target)
      test_loss.backward()
+     test_activations = al.get_log()

  # influence scores for all training data
+ analog.if_all(test_activations)

  # self-influence score
+ analog.self_if(test_activations)
```
