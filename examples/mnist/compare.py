from scipy.stats import pearsonr
import torch


orig = torch.load("if_analog.pt")
lora = torch.load("if_analog_lora64_pca.pt")
print("pearson:", pearsonr(orig, lora))
