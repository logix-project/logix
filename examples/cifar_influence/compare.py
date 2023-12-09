from scipy.stats import pearsonr
import torch


analog_kfac = torch.load("if_baseline.pt")
analog_lora_pca = torch.load("if_analog_pca.pt")
print(
    "[KFAC (analog) vs LoRA-pca (analog)] pearson:",
    pearsonr(analog_kfac, analog_lora_pca),
)
analog_lora_random = torch.load("if_analog_lora.pt")
print(
    "[KFAC (analog) vs LoRA-random (analog)] pearson:",
    pearsonr(analog_kfac, analog_lora_random),
)
