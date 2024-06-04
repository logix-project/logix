import torch
from utils import get_loaders, set_seed

set_seed(0)

# data
_, eval_train_loader, test_loader = get_loaders(
    data_name="sst2",
    valid_indices=list(range(32)),
)

# score
score_path = "if_logix.pt"
scores = torch.load(score_path, map_location="cpu")
print(scores.shape)

for i in range(16):
    print("=" * 80)
    print(f"{i}th data point")
    print(f"Sequence: {test_loader.dataset[i]['sentence']}")
    print(f"Label: {test_loader.dataset[i]['label']}")

    print("Most influential data point")
    rank = torch.argsort(scores[i], descending=True)
    for j in range(3):
        print(f"Rank {j} (score = {scores[i][rank[j]]})")
        print(f"Sentence: {eval_train_loader.dataset[int(rank[j])]['sentence']}")
        print(f"Label: {eval_train_loader.dataset[int(rank[j])]['label']}")
