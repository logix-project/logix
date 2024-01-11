from typing import Dict, Optional, Tuple
import pandas as pd
import torch

from einops import einsum, rearrange, reduce
from analog.utils import get_logger, nested_dict
from analog.analysis import AnalysisBase


class InfluenceFunction(AnalysisBase):
    def __init__(self, config, state):
        super().__init__(config, state)
        self.influence_scores = pd.DataFrame()

    def parse_config(self):
        return

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
        preconditioned = nested_dict()
        (
            covariance_eigval,
            covariance_eigvec,
        ) = self._state.get_covariance_svd_state()
        is_ekfac = hasattr(self._state, "ekfac_eigval_state")
        for module_name in src.keys():
            # if hessian_eigvec is empty, then return src
            if module_name not in covariance_eigvec:
                get_logger().warning(
                    "Hessian has not been computed. No preconditioning applied.\n"
                )
                return src_log

            src_module = src[module_name]["grad"]
            device = src_module.device
            module_eigval = covariance_eigval[module_name]
            module_eigvec = covariance_eigvec[module_name]
            rotated_grad = einsum(
                module_eigvec["backward"].to(device=device).t(),
                src_module,
                module_eigvec["forward"].to(device=device),
                "a b, batch b c, c d -> batch a d",
            )
            scale = (
                module_eigval
                if is_ekfac
                else torch.outer(module_eigval["backward"], module_eigval["forward"])
            ).to(device=device)
            if damping is None:
                damping = 0.1 * torch.mean(scale)
            prec_rotated_grad = rotated_grad / (scale + damping)
            preconditioned[module_name]["grad"] = einsum(
                module_eigvec["backward"].to(device=device),
                prec_rotated_grad,
                module_eigvec["forward"].to(device=device).t(),
                "a b, batch b c, c d -> batch a d",
            )
        return (src_ids, preconditioned)

    @torch.no_grad()
    def dot(self, src: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]):
        """
        Compute the dot product between two gradient dictionaries.

        Args:
            src (Dict[str, torch.Tensor]): Dictionary of source gradients
            tgt (Dict[str, torch.Tensor]): Dictionary of target gradients
        """
        results = 0
        for module_name in src.keys():
            src_module, tgt_module = src[module_name]["grad"], tgt[module_name]["grad"]
            tgt_module = tgt_module.to(device=src_module.device)
            assert src_module.shape[1:] == tgt_module.shape[1:]
            src_module_expanded = rearrange(src_module, "n ... -> n 1 ...")
            tgt_module_expanded = rearrange(tgt_module, "m ... -> 1 m ...")
            module_influence = reduce(
                src_module_expanded * tgt_module_expanded,
                "n m a b -> n m",
                "sum",
            )
            results += module_influence
        return results

    @torch.no_grad()
    def norm(
        self,
        src: Dict[str, torch.Tensor],
        tgt: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Compute the norm of a gradient dictionary.

        Args:
            src (Dict[str, torch.Tensor]): Dictionary of source gradients
            tgt (Optional[Dict[str, torch.Tensor]]): Dictionary of target gradients
        """
        results = 0
        for module_name in src.keys():
            src_module = src[module_name]["grad"]
            tgt_module = tgt[module_name]["grad"] if tgt is not None else src_module
            module_influence = reduce(src_module * tgt_module, "n a b -> n", "sum")
            results += module_influence.reshape(-1)
        return results

    def compute_influence(
        self,
        src_log: Dict[str, Dict[str, torch.Tensor]],
        tgt_log: Dict[str, Dict[str, torch.Tensor]],
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

        # Compute influence scores
        total_influence = None
        if mode == "dot":
            total_influence = self.dot(src, tgt)
        elif mode == "cosine":
            dot = self.dot(src, tgt)
            src_norm = self.norm(src)
            tgt_norm = self.norm(tgt).to(device=src_norm.device)
            total_influence = dot / torch.sqrt(torch.outer(src_norm, tgt_norm))
        elif mode == "l2":
            dot = self.dot(src, tgt)
            src_norm = self.norm(src)
            tgt_norm = self.norm(tgt).to(device=src_norm.device)
            total_influence = 2 * dot - src_norm.unsqueeze(1) - tgt_norm.unsqueeze(0)
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

    def compute_self_influence(
        self,
        src_log: Dict[str, Dict[str, torch.Tensor]],
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
        preconditioned_src = None
        if precondition:
            preconditioned_src = self.precondition(src_log, damping)[1]

        # Compute self-influence scores
        self_influence_scores = self.norm(src, preconditioned_src)

        return self_influence_scores

    def compute_influence_all(
        self,
        src_log: Dict[str, Dict[str, torch.Tensor]],
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
                src_log, tgt_log, mode=mode, precondition=False
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
