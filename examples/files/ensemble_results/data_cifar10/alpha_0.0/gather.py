import argparse
import os
import time

parser = argparse.ArgumentParser("CIFAR Influence Analysis")
parser.add_argument("--algname", type=str)
args = parser.parse_args()

a = 0.0
import torch
for i in range(10):
    t = torch.load(f"{args.algname}{i}.pt")
    a += t

a /= 10
torch.save(a, f"{args.algname}_10.pt")
# delete all the files
for i in range(10):
    os.remove(f"{args.algname}{i}.pt")