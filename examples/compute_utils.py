def get_ensemble_file_name(
    base_path: str, expt_name: str, data_name: str, alpha: str
) -> str:
    return f"{base_path}/data_{data_name}/alpha_{alpha}/{expt_name}.pt"

def get_expt_name_by_config(config, isLora, isEkfac, model_id = None, damping = 1e-10, additional_tag = "", true_fisher = False, use_full_covariance = False, use_augmented_data = False) -> str:
    expt_name = ""
    if isLora:
        expt_name += "lora" + str(config.lora_config["rank"]) + config.lora_config["init"]
    else:
        expt_name += "noLora"

    if not use_full_covariance:
        if isEkfac:
            expt_name += "Ekfac"
        else:
            expt_name += "Kfac"
        
    if damping != 1e-10:
        expt_name += f"_damping{damping}"
    
    if true_fisher:
        expt_name += "_true_fisher"
    
    if use_full_covariance:
        expt_name += "_full_covariance"
    
    if use_augmented_data:
        expt_name += "_aug"

    expt_name += additional_tag

    if model_id is not None:
        expt_name += str(model_id)
    else:
        expt_name += "_10"




    return expt_name