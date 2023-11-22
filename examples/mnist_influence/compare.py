from scipy.stats import pearsonr
import torch


kfac = torch.load("if_kfac.pt")
ekfac = torch.load("if_ekfac.pt")
analog_kfac = torch.load("if_analog.pt")
analog_lora = torch.load("if_analog_lora64_pca.pt")
analog_ekfac = torch.load("if_analog_ekfac.pt")
print("[KFAC (base) vs KFAC (analog)] pearson:", pearsonr(kfac, analog_kfac))
print("[KFAC (base) vs LoRA (analog)] pearson:", pearsonr(kfac, analog_lora))
print("[EKFAC (base) vs EKFAC (analog)] pearson:", pearsonr(ekfac, analog_ekfac))
print("[EKFAC (base) vs KFAC (analog)] pearson:", pearsonr(ekfac, analog_kfac))
print("[EKFAC (base) vs LoRA (analog)] pearson:", pearsonr(ekfac, analog_lora))
