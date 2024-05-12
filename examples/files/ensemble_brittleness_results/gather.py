import os 
import argparse
import numpy as np
import torch
import glob

argparser = argparse.ArgumentParser()
argparser.add_argument("--algo_name", type=str)
argparser.add_argument("--data_name", type=str)

args = argparser.parse_args()
ans = None
algo_name = args.algo_name
data_name = args.data_name

data_path = f"../../files/ensemble_brittleness_results/data_{data_name.lower()}"

# Use glob to find all files starting with algo_name
file_pattern = f"{data_path}/{algo_name}_*.pt"
files = glob.glob(file_pattern)

for file_path in files:
    results = torch.load(file_path)["results"]
    results = np.array(results)
    if ans is None:
        ans = results
    else:
        if not np.any(results == None):
            ans = np.vstack((ans, results))
torch.save({"results": ans},  (f"../../files/ensemble_brittleness_results/data_{data_name.lower()}/{algo_name}.pt"))

#delete
for file_path in files:
    os.remove(file_path)
