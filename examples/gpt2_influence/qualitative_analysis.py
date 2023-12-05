# usage: python qualitative_analysis.py --score_path if_analog.pt --score_path2 files/results/0/wiki_if.pt
import argparse
from transformers import AutoTokenizer

import torch
from scipy.stats import pearsonr, spearmanr

from utils import set_seed, get_loaders


parser = argparse.ArgumentParser("GPT2 Influence Score qualtitative analysis")
parser.add_argument("--score_path", type=str)
parser.add_argument("--score_path2", type=str)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0)


# data
_, eval_train_loader, test_loader = get_loaders(
    valid_indices=list(range(128)),
)

scores = torch.load(args.score_path, map_location="cpu")
if args.score_path2 is not None:
    scores2 = torch.load(args.score_path2, map_location="cpu")

    corr = []
    for s1, s2 in zip(scores, scores2):
        r = pearsonr(s1, s2)[0]
        corr.append(r)
    print(f"Average correlation: {sum(corr) / len(corr)}")

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=True)
for i in range(16):
    print("=" * 80)
    print(f"{i}th data point")
    sequence = tokenizer.decode(test_loader.dataset[i]["input_ids"])
    print(f"Sequence: {sequence}")

    print("Most influential data point")
    rank = torch.argsort(scores[i], descending=True)
    for j in range(3):
        print(f"Rank {j} (score = {scores[i][rank[j]]})")
        sent = tokenizer.decode(eval_train_loader.dataset[int(rank[j])]["input_ids"])
        print(f"Sentence: {sent}")
    input()
