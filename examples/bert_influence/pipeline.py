from typing import Dict

def get_hyperparameters(data_name: str) -> Dict[str, float]:
    wd = 0.0
    if data_name == "rte":
        lr = 2e-5
        epochs = 3
    else:
        raise NotImplementedError()
    return {"lr": lr, "wd": wd, "epochs": epochs}