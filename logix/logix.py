import os

from typing import Optional, Iterable, Dict, Any, List, Union
from dataclasses import asdict
from functools import reduce
from copy import deepcopy

import yaml

import torch
import torch.nn as nn

from logix.analysis import InfluenceFunction
from logix.batch_info import BatchInfo
from logix.config import init_config_from_yaml, LoggingConfig, LoRAConfig
from logix.logging import HookLogger
from logix.logging.log_loader import LogDataset
from logix.logging.log_loader_utils import collate_nested_dicts
from logix.lora import LoRAHandler
from logix.lora.utils import is_lora
from logix.state import LogIXState
from logix.utils import (
    get_logger,
    get_rank,
    get_repr_dim,
    get_world_size,
    print_tracked_modules,
    module_check,
)


class LogIX:
    """
    LogIX is a front-end interface for logging and analyzing neural networks.
    Using (PyTorch) hooks, it tracks, saves, and computes statistics for activations,
    gradients, and other tensors with its logger. Once logging is finished, it provides
    an interface for computing influence scores and other analysis.

    Args:
        project (str):
            The name or identifier of the project. This is used for organizing
            logs and analysis results.
        config (str, optional):
            The path to the YAML configuration file. This file contains settings
            for logging, analysis, and other features. Defaults to an empty string,
            which means default settings are used.
    """

    _SUPPORTED_MODULES = {nn.Linear, nn.Conv1d, nn.Conv2d}

    def __init__(
        self,
        project: str,
        config: Optional[str] = None,
        logging_config: Optional[LoggingConfig] = None,
    ) -> None:
        self.project = project

        self.model = None

        # Config
        self.config = init_config_from_yaml(project=project, logix_config=config)
        if logging_config is not None:
            self.set_logging_config(logging_config)
        self.log_dir = self.config.log_dir

        # LogIX state
        self.state = LogIXState()
        self.binfo = BatchInfo()

        # Initialize logger
        self.logger = HookLogger(
            config=self.get_logging_config(), state=self.state, binfo=self.binfo
        )

        # Log data
        self.log_dataset = None
        self.log_dataloader = None

        # Analysis
        self.influence = InfluenceFunction(state=self.state)

        self.type_filter = None
        self.name_filter = None

    def watch(
        self,
        model: nn.Module,
        type_filter: List[nn.Module] = None,
        name_filter: List[str] = None,
    ) -> None:
        """
        Sets up modules in the model to be watched based on optional type and name filters.
        Hooks will be added to the watched modules to track their forward activations,
        backward error signals, and gradient, and compute various statistics (e.g. mean,
        variance, covariance) for each of these.

        Args:
            model (nn.Module):
                The neural network model to be watched.
            type_filter (List[nn.Module], optional):
                A list of module types to be watched. If None, all supported module types
                are watched.
            name_filter (List[str], optional):
                A list of module names to be watched. If None, all modules are considered.
        """
        self.model = model
        if get_world_size() > 1 and hasattr(self.model, "module"):
            self.model = self.model.module
        self.type_filter = type_filter or self.type_filter
        self.name_filter = name_filter or self.name_filter

        _is_lora = is_lora(self.model)

        for name, module in self.model.named_modules():
            if module_check(
                module=module,
                module_name=name,
                supported_modules=self._SUPPORTED_MODULES,
                type_filter=self.type_filter,
                name_filter=self.name_filter,
                is_lora=_is_lora,
            ):
                self.logger.add_module(name, module)
            elif len(list(module.children())) == 0:
                # disable gradient compute for non-tracked modules. This will save a
                # significant amount of GPU memory and compute for large models
                for p in module.parameters():
                    p.requires_grad = False

        module_path, repr_dim = get_repr_dim(self.logger.modules_to_name)
        print_tracked_modules(reduce(lambda x, y: x + y, repr_dim))
        self.state.set_state("model_module", path=module_path, path_dim=repr_dim)

    def watch_activation(self, tensor_dict: Dict[str, torch.Tensor]) -> None:
        """
        Sets up tensors to be watched. This is useful for logging custom tensors or
        activations that are not directly associated with model modules.

        Args:
            tensor_dict (Dict[str, torch.Tensor]):
                A dictionary where keys are tensor names and values are the tensors
                themselves.
        """
        self.logger.register_all_tensor_hooks(tensor_dict)

    def add_lora(
        self,
        model: Optional[nn.Module] = None,
        watch: bool = True,
        clear: bool = True,
        type_filter: Optional[List[nn.Module]] = None,
        name_filter: Optional[List[str]] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        """
        Adds an LoRA variant for gradient compression. Note that the added LoRA module
        is different from the original LoRA module in that it consists of three parts:
        encoder, bottleneck, and decoder. The encoder and decoder are initialized with
        random weights, while the bottleneck is initialized to be zero. In doing so,
        the encoder and decoder are repsectively used to compress forward and backward
        signals, while the bottleneck is used to extract the compressed gradient.
        Mathematically, this corresponds to random Kronecker product compression.

        Args:
            model (Optional[nn.Module]):
                The model to apply LoRA to. If None, the model already set with watch
                is used.
            watch (bool, optional):
                Whether to watch the model after adding LoRA. Defaults to `True`.
            clear (bool, optional):
                Whether to clear the internal states after adding LoRA. Defaults to
                `True`.
        """
        if lora_config is not None:
            self.set_lora_config(lora_config)

        if not hasattr(self, "lora_handler"):
            self.lora_handler = LoRAHandler(self.get_lora_config(), self.state)

        if model is None:
            model = self.model

        self.lora_handler.add_lora(
            model=model,
            type_filter=type_filter or self.type_filter,
            name_filter=name_filter or self.name_filter,
        )

        # Clear state and logger
        if clear:
            msg = "LogIX will clear the previous Hessian, Storage, and Logging "
            msg += "handlers after adding LoRA for gradient compression.\n"
            get_logger().info(msg)
            self.clear()
        if watch:
            self.watch(model)

    def log(self, data_id: Any, mask: Optional[torch.Tensor] = None) -> None:
        """
        Logs the data. This is an experimental feature for now.

        Args:
            data_id (str):
                A unique identifier associated with the data for the logging session.
            mask (Optional[torch.Tensor], optional):
                An optional attention mask used in Transformer models.
        """
        self.logger.log(data_id=data_id, mask=mask)

    def __call__(
        self,
        data_id: Iterable[Any] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            data_id: A unique identifier associated with the data for the logging session.
            mask (torch.Tensor, optional): Mask for the data.

        Returns:
            self: Returns the instance of the LogIX object.
        """
        self.binfo.clear()
        self.binfo.data_id = data_id
        self.binfo.mask = mask

        return self

    def __enter__(self):
        """
        Sets up the context manager.

        This method is automatically called when the `with` statement is used with an `LogIX` object.
        It sets up the logging environment based on the provided parameters.
        """
        self.logger.clear(hook=True, module=False, buffer=False)
        self.logger.register_all_module_hooks()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Clears the internal states and removes all the hooks.

        This method is essential for ensuring that there are no lingering hooks that could
        interfere with further operations on the model or with future logging sessions.
        """
        self.logger.update()

    def build_log_dataset(self, flatten: bool = False) -> torch.utils.data.Dataset:
        """
        Constructs the log dataset from the stored logs. This dataset can then be used
        for analysis or visualization.

        Returns:
            LogDataset:
                An instance of LogDataset containing the logged data.
        """
        if self.log_dataset is None:
            self.log_dataset = LogDataset(log_dir=self.log_dir, flatten=flatten)
        return self.log_dataset

    def build_log_dataloader(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        flatten: bool = False,
    ) -> torch.utils.data.DataLoader:
        """
        Constructs a DataLoader for the log dataset. This is useful for batch processing
        of logged data during analysis.

        Return:
            DataLoader:
                A DataLoader instance for the log dataset.
        """
        if self.log_dataloader is None:
            log_dataset = self.build_log_dataset(flatten=flatten)
            collate_fn = None if flatten else collate_nested_dicts
            self.log_dataloader = torch.utils.data.DataLoader(
                log_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                collate_fn=collate_fn,
            )
        return self.log_dataloader

    def get_log(self, copy=False) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns the current log, including data identifiers and logged information.

        Returns:
            dict: The current log.
        """
        log = (self.binfo.data_id, self.binfo.log)
        return log if not copy else deepcopy(log)

    def get_covariance_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns the covariance state from the Hessian handler.
        """
        return self.state.get_covariance_state()

    def get_covariance_svd_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns the SVD of the covariance from the Hessian handler.
        """
        return self.state.get_covariance_svd_state()

    def compute_influence(
        self,
        src_log: Dict[str, Dict[str, torch.Tensor]],
        tgt_log: Dict[str, Dict[str, torch.Tensor]],
        mode: Optional[str] = "dot",
        precondition: Optional[bool] = True,
    ) -> Dict[str, Union[List[str], torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Front-end interface for computing influence scores. It calls the
        `compute_influence` method of the `InfluenceFunction` class.

        Args:
            src_log (Tuple[str, Dict[str, Dict[str, torch.Tensor]]]): Log of source gradients
            tgt_log (Tuple[str, Dict[str, Dict[str, torch.Tensor]]]): Log of target gradients
            mode (str, optional): Influence function mode. Defaults to "dot".
            precondition (bool, optional): Whether to precondition the gradients. Defaults to True.
        """
        return self.influence.compute_influence(
            src_log, tgt_log, mode=mode, precondition=precondition
        )

    def compute_influence_all(
        self,
        src_log: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        loader: Optional[torch.utils.data.DataLoader] = None,
        mode: Optional[str] = "dot",
        precondition: Optional[str] = True,
    ) -> Dict[str, Union[List[str], torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Front-end interface for computing influence scores against all train data in the log.
        It calls the `compute_influence_all` method of the `InfluenceFunction` class.

        Args:
            src_log (Tuple[str, Dict[str, Dict[str, torch.Tensor]]]): Log of source gradients
            loader (torch.utils.data.DataLoader): DataLoader of train data
            mode (str, optional): Influence function mode. Defaults to "dot".
            precondition (bool, optional): Whether to precondition the gradients. Defaults to True.
        """
        src_log = src_log if src_log is not None else self.get_log()
        loader = loader if loader is not None else self.build_log_dataloader()
        return self.influence.compute_influence_all(
            src_log, loader, mode=mode, precondition=precondition
        )

    def compute_self_influence(
        self,
        src_log: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        precondition: Optional[bool] = True,
    ) -> Dict[str, Union[List[str], torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Front-end interface for computing self-influence scores. It calls the
        `compute_self_influence` method of the `InfluenceFunction` class.

        Args:
            src_log (Tuple[str, Dict[str, Dict[str, torch.Tensor]]]): Log of source gradients
            precondition (bool, optional): Whether to precondition the gradients. Defaults to True.
        """
        src_log = src_log if src_log is not None else self.get_log()
        return self.influence.compute_self_influence(src_log, precondition=precondition)

    def save_config(self) -> None:
        """
        Save LogIX state to disk.
        """
        config_file = os.path.join(self.log_dir, "config.yaml")
        config_dict = asdict(self.config)
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def save_state(self) -> None:
        """
        Save Hessian state to disk.
        """
        self.state.save_state(self.log_dir)

    def save_lora(self) -> None:
        """
        Save LoRA state to disk.
        """
        state_dict = self.model.state_dict()
        lora_state_dict = {
            name: param for name, param in state_dict.items() if "logix_lora" in name
        }
        if len(lora_state_dict) > 0:
            log_dir = os.path.join(self.log_dir, "lora")
            if not os.path.exists(log_dir) and get_rank() == 0:
                os.makedirs(log_dir)
            torch.save(lora_state_dict, os.path.join(log_dir, "lora_state_dict.pt"))

    def initialize_from_log(self) -> None:
        """
        Load all states from disk.
        """
        # Load logix config
        assert os.path.exists(self.log_dir), f"{self.log_dir} does not exist!"

        config_file = os.path.join(self.log_dir, "config.yaml")
        self.config.load_config(config_file)

        # Load LoRA state
        lora_dir = os.path.join(self.log_dir, "lora")
        if os.path.exists(lora_dir) and self.model is not None:
            if not is_lora(self.model):
                self.add_lora()
            lora_state = torch.load(os.path.join(lora_dir, "lora_state_dict.pt"))
            for name in lora_state:
                assert name in self.model.state_dict(), f"{name} not in model!"
            self.model.load_state_dict(lora_state, strict=False)

        # Load state
        self.state.load_state(self.log_dir)

    def finalize(
        self,
    ) -> None:
        """
        Finalizes the logging session.

        Args:
            clear (bool, optional): Whether to clear the internal states or not.
        """
        self.state.finalize()
        self.logger.finalize()

        if get_rank() == 0:
            self.save_config()
            self.save_state()
            self.save_lora()

    def setup(self, log_option_kwargs: Dict[str, Any]) -> None:
        """
        Update logging configurations.

        Args:
            log_option_kwargs: Logging configurations.
        """
        self.logger.opt.setup(log_option_kwargs)

    def eval(self) -> None:
        """
        Set the state of LogIX for testing.
        """
        self.logger.opt.eval()

    def clear(self) -> None:
        """
        Clear everything in LogIX.
        """
        self.state.clear()
        self.logger.clear()

    def set_logging_config(self, logging_config: LoggingConfig) -> None:
        logging_config.log_dir = self.config.log_dir
        self.config.logging = logging_config

    def set_lora_config(self, lora_config: LoRAConfig) -> None:
        self.config.lora = lora_config

    def get_logging_config(self) -> LoggingConfig:
        return self.config.logging

    def get_lora_config(self) -> LoRAConfig:
        return self.config.lora
