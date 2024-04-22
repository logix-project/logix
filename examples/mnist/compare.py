from scipy.stats import pearsonr
import torch


kfac = torch.load("if_kfac.pt")
kfac_true = torch.load("if_kfac_true.pt")
ekfac = torch.load("if_ekfac.pt")
ekfac_true = torch.load("if_ekfac_true.pt")
logix_kfac = torch.load("if_logix.pt")
logix_lora = torch.load("if_logix_lora64_pca.pt")
logix_ekfac = torch.load("if_logix_ekfac.pt")
print("[KFAC (base) vs KFAC (logix)] pearson:", pearsonr(kfac, logix_kfac))
print("[KFAC (base) vs LoRA (logix)] pearson:", pearsonr(kfac, logix_lora))
print("[EKFAC (base) vs EKFAC (logix)] pearson:", pearsonr(ekfac, logix_ekfac))
print("[EKFAC (base) vs KFAC (logix)] pearson:", pearsonr(ekfac, logix_kfac))
print("[EKFAC (base) vs LoRA (logix)] pearson:", pearsonr(ekfac, logix_lora))
