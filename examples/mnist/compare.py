from scipy.stats import pearsonr
import torch


orig = torch.load("if_analog.pt")
lora = torch.load("if_analog_lora64_pca.pt")
kfac = torch.load("if_kfac.pt")
print("[Orig vs LoRA] pearson:", pearsonr(orig, lora))
print("[KFAC vs Orig] pearson:", pearsonr(kfac, orig))
print("[KFAC vs LoRA] pearson:", pearsonr(kfac, lora))
