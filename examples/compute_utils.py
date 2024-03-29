from analog.analog import Config

def get_ensemble_file_name(
    base_path: str, expt_name: str, data_name: str, alpha: str
) -> str:
    return f"{base_path}/data_{data_name}/alpha_{alpha}/{expt_name}.pt"

def get_expt_name_by_config(analog_config: Config, argparse_args) -> str:
    expt_name = ""
    for key, value in analog_config.lora_config.items():
        expt_name += f"{key}{value}_"

    for key, value in analog_config.logging_options.items():
        expt_name += f"{key}{value}_"
    
    #add addparser args
    argparse_args_excludes = ["data", "resume"]
    for key, value in vars(argparse_args).items():
        if key not in argparse_args_excludes:
            expt_name += f"{key}{value}_"

    return expt_name[:-1]