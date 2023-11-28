from scipy.stats import pearsonr
import torch


kfac = torch.load("if_kfac.pt")
kfac_true = torch.load("if_kfac_true.pt")
ekfac = torch.load("if_ekfac.pt")
ekfac_true = torch.load("if_ekfac_true.pt")
analog_kfac = torch.load("if_analog.pt")
analog_lora = torch.load("if_analog_lora64_pca.pt")
analog_ekfac = torch.load("if_analog_ekfac.pt")
analog_scheduler = torch.load("if_analog_scheduler.pt")
print("[KFAC (base) vs KFAC (analog)] pearson:", pearsonr(kfac, analog_kfac))
print("[KFAC (base) vs LoRA (analog)] pearson:", pearsonr(kfac, analog_lora))
print("[EKFAC (base) vs EKFAC (analog)] pearson:", pearsonr(ekfac, analog_ekfac))
print("[EKFAC (base) vs KFAC (analog)] pearson:", pearsonr(ekfac, analog_kfac))
print("[EKFAC (base) vs LoRA (analog)] pearson:", pearsonr(ekfac, analog_lora))
print(
    "[Scheduler (analog) vs EKFAC (analog)] pearson:",
    pearsonr(analog_scheduler, analog_ekfac),
)
print(
    "[Scheduler (analog) vs KFAC (analog)] pearson:",
    pearsonr(analog_scheduler, analog_kfac),
)
print(
    "[Scheduler (analog) vs LoRA (analog)] pearson:",
    pearsonr(analog_scheduler, analog_lora),
)
print(
    "[Scheduler (analog) vs KFAC_TRUE (base)] pearson:",
    pearsonr(analog_scheduler, kfac_true),
)
print(
    "[Scheduler (analog) vs EKAFC_TRUE (base)] pearson:",
    pearsonr(analog_scheduler, ekfac_true),
)
