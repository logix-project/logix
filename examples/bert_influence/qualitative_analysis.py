# usage: python qualitative_analysis.py --score_path files/analog/lora_random_128_d1e-05/if_analog.pt --score_path2 files/results/0/sst2_ekfac_empirical_d1e-5_32_if.pt

import argparse

import torch
from scipy.stats import pearsonr

from utils import get_loaders, set_seed


parser = argparse.ArgumentParser("GPT2 Influence Score qualtitative analysis")
parser.add_argument("--score_path", type=str)
parser.add_argument("--score_path2", type=str)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

# data
_, eval_train_loader, test_loader = get_loaders(
    data_name="sst2",
    valid_indices=list(range(32)),
)

# correlation
scores = torch.load(args.score_path, map_location="cpu")
print(f"Loaded scores from {args.score_path}")
if args.score_path2 is not None:
    scores2 = torch.load(args.score_path2, map_location="cpu")
    print(f"Loaded scores from {args.score_path2}")
    print(f"scores: {scores.shape}")
    print(f"scores2: {scores2.shape}")

    corr = []
    for s1, s2 in zip(scores, scores2):
        r = pearsonr(s1, s2)[0]
        corr.append(r)
    print(f"Average correlation: {sum(corr) / len(corr)}")
