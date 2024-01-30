def get_ensemble_file_name(
    base_path: str, expt_name: str, data_name: str, alpha: str
) -> str:
    return f"{base_path}/data_{data_name}/alpha_{alpha}/{expt_name}.pt"

def get_expt_name_by_loraConfig(args, lora_config) -> str:
    expt_name = ("lora" if args.lora else "noLora") + ("Ekfac" if args.ekfac else "Kfac") + lora_config["init"] + str(lora_config["rank"]) + str(args.model_id)
    return expt_name