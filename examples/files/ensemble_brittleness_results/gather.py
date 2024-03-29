import os 
import argparse
import numpy as np
import torch


argparser = argparse.ArgumentParser()
argparser.add_argument("--algo_name", type=str)
argparser.add_argument("--data_name", type=str)

args = argparser.parse_args()
ans = None
algo_name = args.algo_name
data_name = args.data_name
for startIdx in range(0,100,10):
    results = torch.load(
        f"../../files/ensemble_brittleness_results/data_{data_name.lower()}/{algo_name}_{startIdx}_{startIdx+10}.pt"
    )["results"]
    results = np.array(results)
    if ans is None:
        ans = results
    else:
        ans = np.vstack((ans, results))
torch.save({"results": ans},  (f"../../files/ensemble_brittleness_results/data_{data_name.lower()}/{algo_name}.pt"))

#delete
for startIdx in range(0,100,10):
    os.remove(f"../../files/ensemble_brittleness_results/data_{data_name.lower()}/{algo_name}_{startIdx}_{startIdx+10}.pt")