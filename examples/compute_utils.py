from dataclasses import asdict
from logix.config import Config

def get_ensemble_file_name(
    base_path: str, expt_name: str, data_name: str, alpha: str
) -> str:
    return f"{base_path}/data_{data_name}/alpha_{alpha}/{expt_name}.pt"

def get_expt_name_by_config(logix_config: Config, argparse_args) -> str:
    expt_name = ""
    for key, value in asdict(logix_config.lora).items():
        if value is not False and value is not None:
            expt_name += f"{key}{value}_"

    for key, value in asdict(logix_config.scheduler).items():
        if value is not False and value is not None:
            expt_name += f"{key}{value}_"
    
    #add addparser args
    argparse_args_excludes = ["data", "resume", "startIdx", "endIdx", "scoreFileName"]
    for key, value in vars(argparse_args).items():
        if key not in argparse_args_excludes and value is not False:
            expt_name += f"{key}{value}_"

    return expt_name[:-1]