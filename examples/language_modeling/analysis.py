import os

import torch


k = 20
experiment = "pythia-1.4b_random_raw_openwebtext_mlp/generated"
mode = "cosine"

scores = torch.load(os.path.join(experiment, f"scores_{mode}.pt"), map_location="cpu")
train_ids = torch.load(
    os.path.join(experiment, f"train_ids_{mode}.pt"), map_location="cpu"
)
test_ids = torch.load(
    os.path.join(experiment, f"test_ids_{mode}.pt"), map_location="cpu"
)
print(len(train_ids), len(test_ids), scores.shape)
assert len(train_ids) == scores.shape[1]
assert len(test_ids) == scores.shape[0]

out = ""
for idx, test_id in enumerate(test_ids):
    out += "==========================================================\n"
    out += f"Query: {test_id}\n"
    out += "==========================================================\n"
    topk_indices = torch.topk(scores[idx], k=k)[1]
    for j, topk_idx in enumerate(topk_indices):
        score = scores[idx][topk_idx]
        train_id = train_ids[topk_idx]
        out += f"Top {j + 1} (score: {score})]: {train_id}\n"
        out += "==========================================================\n"

with open(os.path.join(experiment, f"top_{mode}.txt"), "w") as file:
    file.write(out)
