import torch


x = torch.load("data_0.pt")

for k, v in x.items():
    assert len(v) == 3

    for k1, v1 in v.items():
        assert len(v1) == 2

        for k2, v2 in v1.items():
            print(v2.shape)
