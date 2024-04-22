from scipy.stats import pearsonr
import torch


logix_kfac = torch.load("if_baseline.pt")
logix_lora_pca = torch.load("if_logix_pca.pt")
print(
    "[KFAC (logix) vs LoRA-pca (logix)] pearson:",
    pearsonr(logix_kfac, logix_lora_pca),
)
logix_lora_random = torch.load("if_logix_lora.pt")
print(
    "[KFAC (logix) vs LoRA-random (logix)] pearson:",
    pearsonr(logix_kfac, logix_lora_random),
)
