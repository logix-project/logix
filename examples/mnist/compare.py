from scipy.stats import pearsonr
import torch


kfac = torch.load("if_kfac.pt")
ekfac = torch.load("if_ekfac.pt")
logix = torch.load("if_logix.pt")
print("[KFAC (base) vs LogIX] pearson:", pearsonr(kfac, logix))
print("[EKFAC (base) vs LogIX] pearson:", pearsonr(ekfac, logix))
