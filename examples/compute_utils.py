def get_ensemble_file_name(
    base_path: str, expt_name: str, data_name: str, alpha: str
) -> str:
    return f"{base_path}/data_{data_name}/alpha_{alpha}/{expt_name}.pt"

def get_expt_name_by_config(config, isLora, isEkfac, model_id = None, damping = 1e-10) -> str:
    expt_name = ("lora" if isLora else "noLora") \
    + ("Ekfac" if isEkfac else "Kfac") \
    + config.lora_config["init"] \
    + str(config.lora_config["rank"]) \
    + ( ("_damping" + str(damping)) if damping != 1e-10 else "")\
    + (str(model_id) if model_id is not None else "_10") 
    return expt_name