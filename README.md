# AnaLog

### What is AnaLog?
AnaLog is a scalable and interoperable machine learning debugging tool. It's built upon the asssumption that training logs (e.g. activations) have rich information about the final model, so debugging can be facilitated by analyzing these logs.

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

 ''' Your Training Code '''

+ analog = AnaLog(project="analog")
+ analog.watch(model)
  for input, target in data_loader:
+     with analog(data_id=input, log="activation", hessian=True, save=True):
          out = model(input)
          loss = loss_fn(out, target)
          loss.backward()
          model.zero_grad()
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
