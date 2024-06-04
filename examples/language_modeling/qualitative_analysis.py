# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer
from utils import get_loaders, set_seed

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
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=True)

scores = torch.load(args.score_path, map_location="cpu")
if args.score_path2 is not None:
    scores2 = torch.load(args.score_path2, map_location="cpu")

    corr = []
    for s1, s2 in zip(scores, scores2):
        r = pearsonr(s1, s2)[0]
        corr.append(r)
    print(f"Average correlation: {sum(corr) / len(corr)}")

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
