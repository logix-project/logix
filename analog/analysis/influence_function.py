from typing import Dict, Optional, Tuple
import pandas as pd
import torch

from einops import einsum, rearrange, reduce
from analog.config import InfluenceConfig
from analog.state import AnaLogState
from analog.utils import get_logger, nested_dict
from analog.analysis.utils import synchronize_device


class InfluenceFunction:
    def __init__(self, config: InfluenceConfig, state: AnaLogState):
        # state
        self._state = state

        # config
        self.log_dir = config.log_dir
        self.mode = config.mode
        self.damping = config.damping
        self.relative_damping = config.relative_damping

        # influence scores
        self.influence_scores = pd.DataFrame()
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
        cov_eigval, cov_eigvec = self._state.get_covariance_svd_state()
        if set(cov_eigvec.keys()) != set(src.keys()):
            get_logger().warning(
                "Not all covariances have been computed. No preconditioning applied.\n"
            )
            return src_log

        preconditioned = nested_dict()
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
            damping = damping or self.damping
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
        total_influence = total_influence.cpu()

        # Log influence scores to pd.DataFrame
        assert total_influence.shape[0] == len(src_ids)
        assert total_influence.shape[1] == len(tgt_ids)
        # Ensure src_ids and tgt_ids are in the DataFrame's index and columns
        self.influence_scores = self.influence_scores.reindex(
            index=self.influence_scores.index.union(src_ids),
            columns=self.influence_scores.columns.union(tgt_ids),
        )

        # Assign total_influence values to the corresponding locations
        src_indices = [
            self.influence_scores.index.get_loc(src_id) for src_id in src_ids
        ]
        tgt_indices = [
            self.influence_scores.columns.get_loc(tgt_id) for tgt_id in tgt_ids
        ]

        self.influence_scores.iloc[src_indices, tgt_indices] = total_influence.numpy()

        return total_influence

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
        src = src_log[1]
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

        return total_influence

    def flatten_log(self, src):
        to_cat = []
        for module, log_type in self._state.get_state("model_module")["path"]:
            log = src[module][log_type]
            bsz = log.shape[0]
            to_cat.append(log.view(bsz, -1))
        return torch.cat(to_cat, dim=1)

    def compute_influence_all(
        self,
        src_log: Tuple[str, Dict[str, Dict[str, torch.Tensor]]],
        loader: torch.utils.data.DataLoader,
        mode: Optional[str] = "dot",
        precondition: Optional[bool] = True,
        damping: Optional[float] = None,
    ):
        """
        Compute influence scores against all train data in the log. This can be used
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

        if_scores_total = []
        for tgt_log in loader:
            if_scores = self.compute_influence(
                src_log, tgt_log, mode=mode, precondition=False, damping=damping
            )
            if_scores_total.append(if_scores)
        return torch.cat(if_scores_total, dim=-1)

    def get_influence_scores(self):
        """
        Return influence scores as a pd.DataFrame.
        """
        return self.influence_scores

    def save_influence_scores(self, filename="influence_scores.csv"):
        """
        Save influence scores as a csv file.

        Args:
            filename (str, optional): save filename. Defaults to "influence_scores.csv".
        """
        self.influence_scores.to_csv(filename, index=True, header=True)
        get_logger().info(f"Influence scores saved to {filename}")
