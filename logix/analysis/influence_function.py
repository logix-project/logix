from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import torch

from einops import reduce
from logix.config import InfluenceConfig
from logix.state import LogIXState
from logix.utils import (
    get_logger,
    nested_dict,
    flatten_log,
    unflatten_log,
    synchronize_device,
)
from logix.analysis.influence_function_utils import (
    precondition_kfac,
    precondition_raw,
    cross_dot_product,
)
from logix.statistic.utils import make_2d


class InfluenceFunction:
    def __init__(self, config: InfluenceConfig, state: LogIXState):
        # state
        self._state = state

        # config
        self.log_dir = config.log_dir
        self.mode = config.mode
        self.damping = config.damping
        self.relative_damping = config.relative_damping
        self.flatten = config.flatten

    @torch.no_grad()
    def precondition(
        self,
        src_log: Dict[str, Dict[str, torch.Tensor]],
        damping: Optional[float] = None,
        hessian: Optional[str] = "auto",
    ) -> Tuple[List[str], Dict[str, Dict[str, torch.Tensor]]]:
        """
        Precondition gradients using the Hessian.

        Args:
            src_log (Dict[str, Dict[str, torch.Tensor]]): Log of source gradients
            damping (Optional[float], optional): Damping parameter for preconditioning. Defaults to None.
        """
        assert hessian in ["auto", "kfac", "raw"]

        src_ids, src = src_log
        cov_state = self._state.get_covariance_state()
        if len(set(src.keys()) - set(cov_state.keys())) != 0:
            get_logger().warning(
                "Not all covariances have been computed. No preconditioning applied.\n"
            )
            return src_log

        precondition_fn = precondition_kfac
        if hessian == "raw" or (
            hessian == "auto" and "grad" in cov_state[list(src.keys())[0]]
        ):
            precondition_fn = precondition_raw
        preconditioned_grad = precondition_fn(
            src=src, state=self._state, damping=damping
        )

        return (src_ids, preconditioned_grad)

    @torch.no_grad()
    def compute_influence(
        self,
        src_log: Tuple[str, Dict[str, Dict[str, torch.Tensor]]],
        tgt_log: Tuple[str, Dict[str, Dict[str, torch.Tensor]]],
        mode: Optional[str] = "dot",
        precondition: Optional[bool] = True,
        precondition_hessian: Optional[str] = "auto",
        damping: Optional[float] = None,
    ):
        """
        Compute influence scores between two gradient dictionaries.

        Args:
            src_log (Dict[str, Dict[str, torch.Tensor]]): Log of source gradients
            tgt_log (Dict[str, Dict[str, torch.Tensor]]): Log of target gradients
            mode (Optional[str], optional): Influence function mode. Defaults to "dot".
            precondition (Optional[bool], optional): Whether to precondition the gradients. Defaults to True.
            damping (Optional[float], optional): Damping parameter for preconditioning. Defaults to None.
        """
        assert mode in ["dot", "l2", "cosine"], f"Invalid mode: {mode}"

        result = {}
        if precondition:
            src_log = self.precondition(
                src_log=src_log, damping=damping, hessian=precondition_hessian
            )

        src_ids, src = src_log
        tgt_ids, tgt = tgt_log
        total_influence = 0

        # Compute influence scores. By default, we should compute the basic influence
        # scores, which is essentially the inner product between the source and target
        # gradients. If mode is cosine, we should normalize the influence score by the
        # L2 norm of the target gardients. If mode is l2, we should subtract the L2
        # norm of the target gradients.
        if self.flatten:
            src = flatten_log(
                log=src, path=self._state.get_state("model_module")["path"]
            )
            tgt = tgt.to(device=src.device)
            total_influence += cross_dot_product(src, tgt)
        else:
            synchronize_device(src, tgt)
            for module_name in src.keys():
                total_influence += cross_dot_product(
                    src[module_name]["grad"], tgt[module_name]["grad"]
                )

        if mode == "cosine":
            tgt_norm = self.compute_self_influence(
                tgt_log, precondition=True, damping=damping
            )
            total_influence /= torch.sqrt(tgt_norm.unsqueeze(0))
        elif mode == "l2":
            tgt_norm = self.compute_self_influence(
                tgt_log, precondition=True, damping=damping
            )
            total_influence = 2 * total_influence - tgt_norm.unsqueeze(0)

        assert total_influence.shape[0] == len(src_ids)
        assert total_influence.shape[1] == len(tgt_ids)

        result["src_ids"] = src_ids
        result["tgt_ids"] = tgt_ids
        result["influence"] = total_influence.cpu()

        return result

    @torch.no_grad()
    def compute_self_influence(
        self,
        src_log: Tuple[str, Dict[str, Dict[str, torch.Tensor]]],
        precondition: Optional[bool] = True,
        damping: Optional[float] = None,
    ):
        """
        Compute self-influence scores. This can be used for uncertainty estimation.

        Args:
            src_log (Dict[str, Dict[str, torch.Tensor]]): Log of source gradients
            precondition (Optional[bool], optional): Whether to precondition the gradients. Defaults to True.
            damping (Optional[float], optional): Damping parameter for preconditioning. Defaults to None.
        """
        result = {}

        src_ids, src = src_log
        if not isinstance(src, dict):
            assert isinstance(src, torch.Tensor)
            src = unflatten_log(
                log=src, path=self._state.get_state("model_module")["path"]
            )
        tgt = self.precondition(src_log, damping)[1] if precondition else src

        # Compute self-influence scores
        total_influence = 0
        for module_name in src.keys():
            src_module = src[module_name]["grad"]
            tgt_module = tgt[module_name]["grad"] if tgt is not None else src_module
            module_influence = reduce(src_module * tgt_module, "n a b -> n", "sum")
            total_influence += module_influence.reshape(-1)

        result["src_ids"] = src_ids
        result["influence"] = total_influence.cpu()

        return result

    def compute_influence_all(
        self,
        src_log: Tuple[str, Dict[str, Dict[str, torch.Tensor]]],
        loader: torch.utils.data.DataLoader,
        mode: Optional[str] = "dot",
        precondition: Optional[bool] = True,
        damping: Optional[float] = None,
    ):
        """
        Compute influence scores against all traininig data in the log. This can be used
        for training data attribution.

        Args:
            src_log (Dict[str, Dict[str, torch.Tensor]]): Log of source gradients
            loader (torch.utils.data.DataLoader): DataLoader of train data
            mode (Optional[str], optional): Influence function mode. Defaults to "dot".
            precondition (Optional[bool], optional): Whether to precondition the gradients. Defaults to True.
            damping (Optional[float], optional): Damping parameter for preconditioning. Defaults to None.
        """
        if precondition:
            src_log = self.precondition(src_log, damping)

        result_all = None
        for tgt_log in tqdm(loader, desc="Compute IF"):
            result = self.compute_influence(
                src_log, tgt_log, mode=mode, precondition=False, damping=damping
            )

            # Merge results
            if result_all is None:
                result_all = result
            else:
                result_all["tgt_ids"].extend(result["tgt_ids"])
                result_all["influence"] = torch.cat(
                    [result_all["influence"], result["influence"]], dim=1
                )

        return result_all
