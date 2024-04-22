from typing import Dict, Optional, Tuple
from tqdm import tqdm
import torch

from einops import einsum, rearrange, reduce
from logix.config import InfluenceConfig
from logix.state import LogIXState
from logix.utils import get_logger, nested_dict
from logix.analysis.utils import synchronize_device
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
    ):
        """
        Precondition gradients using the Hessian.

        Args:
            src_log (Dict[str, Dict[str, torch.Tensor]]): Log of source gradients
            damping (Optional[float], optional): Damping parameter for preconditioning. Defaults to None.
        """
        src_ids, src = src_log
        cov_state = self._state.get_covariance_state()
        if len(set(src.keys()) - set(cov_state.keys())) != 0:
            get_logger().warning(
                "Not all covariances have been computed. No preconditioning applied.\n"
            )
            return src_log

        preconditioned = nested_dict()
        if "grad" in cov_state[list(src.keys())[0]]:
            cov_inverse = self._state.get_covariance_inverse_state()
            for module_name in src.keys():
                device = src[module_name]["grad"].device
                grad_cov_inverse = cov_inverse[module_name]["grad"].to(device=device)
                original_shape = src[module_name]["grad"].shape
                preconditioned[module_name]["grad"] = (
                    make_2d(src[module_name]["grad"], None, "grad") @ grad_cov_inverse
                ).reshape(original_shape)
        else:
            cov_eigval, cov_eigvec = self._state.get_covariance_svd_state()
            for module_name in src.keys():
                src_grad = src[module_name]["grad"]
                device = src_grad.device

                module_eigvec = cov_eigvec[module_name]
                fwd_eigvec = module_eigvec["forward"].to(device=device)
                bwd_eigvec = module_eigvec["backward"].to(device=device)

                # Reconstruct the full eigenvalue matrix with the damping factor added
                module_eigval = cov_eigval[module_name]
                if isinstance(module_eigval, torch.Tensor):
                    full_eigval = module_eigval.to(device=device)
                else:
                    assert "forward" in module_eigval and "backward" in module_eigval
                    fwd_eigval = module_eigval["forward"]
                    bwd_eigval = module_eigval["backward"]
                    full_eigval = torch.outer(bwd_eigval, fwd_eigval).to(device=device)
                if damping is None:
                    damping = 0.1 * torch.mean(full_eigval)
                full_eigval += damping

                # Precondition the gradient using eigenvectors and eigenvalues
                rotated_grad = einsum(
                    bwd_eigvec.t(),
                    src_grad,
                    fwd_eigvec,
                    "a b, batch b c, c d -> batch a d",
                )
                prec_rotated_grad = rotated_grad / full_eigval
                preconditioned[module_name]["grad"] = einsum(
                    bwd_eigvec,
                    prec_rotated_grad,
                    fwd_eigvec.t(),
                    "a b, batch b c, c d -> batch a d",
                )
        return (src_ids, preconditioned)

    @torch.no_grad()
    def compute_influence(
        self,
        src_log: Tuple[str, Dict[str, Dict[str, torch.Tensor]]],
        tgt_log: Tuple[str, Dict[str, Dict[str, torch.Tensor]]],
        mode: Optional[str] = "dot",
        precondition: Optional[bool] = True,
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
            src_log = self.precondition(src_log, damping)

        src_ids, src = src_log
        tgt_ids, tgt = tgt_log
        total_influence = 0

        if self.flatten:
            src = self.flatten_log(src)
            tgt = tgt.to(device=src.device)
            total_influence += self._dot_product_logs(src, tgt)

        if not self.flatten:
            synchronize_device(src, tgt)
            # Compute influence scores. By default, we should compute the basic influence
            # scores, which is essentially the inner product between the source and target
            # gradients. If mode is cosine, we should normalize the influence score by the
            # L2 norm of the target gardients. If mode is l2, we should subtract the L2
            # norm of the target gradients.
            for module_name in src.keys():
                total_influence += self._dot_product_logs(
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

    def _dot_product_logs(self, src_module, tgt_module):
        assert src_module.shape[1:] == tgt_module.shape[1:]
        src_module_expanded = rearrange(src_module, "n ... -> n 1 ...")
        tgt_module_expanded = rearrange(tgt_module, "m ... -> 1 m ...")
        return reduce(
            src_module_expanded * tgt_module_expanded,
            "n m ... -> n m",
            "sum",
        )

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
        tgt = self.precondition(src_log, damping)[1] if precondition else src

        # Compute self-influence scores
        total_influence = 0

        if self.flatten:
            src = self.flatten_log(src)
            tgt = self.flatten_log(tgt)
            total_influence += self._dot_product_logs(src, tgt)

        for module_name in src.keys():
            src_module = src[module_name]["grad"]
            tgt_module = tgt[module_name]["grad"] if tgt is not None else src_module
            module_influence = reduce(src_module * tgt_module, "n a b -> n", "sum")
            total_influence += module_influence.reshape(-1)

        result["src_ids"] = src_ids
        result["influence"] = total_influence.cpu()

        return result

    def flatten_log(self, src):
        flat_log_list = []
        for module, log_type in self._state.get_state("model_module")["path"]:
            log = src[module][log_type]
            bsz = log.shape[0]
            flat_log_list.append(log.view(bsz, -1))
        flat_log = torch.cat(flat_log_list, dim=1)

        return flat_log

    def compute_influence_all(
        self,
        src_log: Tuple[str, Dict[str, Dict[str, torch.Tensor]]],
        loader: torch.utils.data.DataLoader,
        mode: Optional[str] = "dot",
        precondition: Optional[bool] = True,
        damping: Optional[float] = None,
    ):
        """
        Compute influence scores against all train
        ata in the log. This can be used
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
