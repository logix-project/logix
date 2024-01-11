import pandas as pd
import torch

from einops import einsum, rearrange, reduce
from analog.utils import get_logger
from analog.analysis import AnalysisBase
from analog.analysis.utils import reconstruct_grad, do_decompose, rescaled_dot_product


class InfluenceFunction(AnalysisBase):
    def __init__(self, config, state):
        super().__init__(config, state)
        self.influence_scores = pd.DataFrame()

    def parse_config(self):
        return

    @torch.no_grad()
    def precondition(self, src_log, damping=None):
        src_ids, src = src_log
        preconditioned = {}
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
            preconditioned[module_name] = einsum(
                module_eigvec["backward"].to(device=device),
                prec_rotated_grad,
                module_eigvec["forward"].to(device=device).t(),
                "a b, batch b c, c d -> batch a d",
            )
        return (src_ids, preconditioned)

    @torch.no_grad()
    def compute_influence(self, src_log, tgt_log, precondition=True, damping=None):
        if precondition:
            src_log = self.precondition(src_log, damping)
        src_ids, src = src_log
        tgt_ids, tgt = tgt_log

        # Compute influence scores
        total_influence = 0.0
        for module_name in src.keys():
            src_module, tgt_module = src[module_name], tgt[module_name]["grad"]
            tgt_module = tgt_module.to(device=src_module.device)
            assert src_module.shape[1:] == tgt_module.shape[1:]
            src_module_expanded = rearrange(src_module, "n ... -> n 1 ...")
            tgt_module_expanded = rearrange(tgt_module, "m ... -> 1 m ...")
            module_influence = reduce(
                src_module_expanded * tgt_module_expanded,
                "n m a b -> n m",
                "sum",
            )
            total_influence += module_influence

        # Log influence scores to pd.DataFrame
        assert total_influence.shape[0] == len(src_ids)
        assert total_influence.shape[1] == len(tgt_ids)
        # Ensure src_ids and tgt_ids are in the DataFrame's index and columns, respectively
        self.influence_scores = self.influence_scores.reindex(
            index=self.influence_scores.index.union(src_ids),
            columns=self.influence_scores.columns.union(tgt_ids),
        )

        # Assign total_influence values to the corresponding locations in influence_scores
        src_indices = [
            self.influence_scores.index.get_loc(src_id) for src_id in src_ids
        ]
        tgt_indices = [
            self.influence_scores.columns.get_loc(tgt_id) for tgt_id in tgt_ids
        ]

        self.influence_scores.iloc[
            src_indices, tgt_indices
        ] = total_influence.cpu().numpy()

        return total_influence

    @torch.no_grad()
    def compute_self_influence(self, src_log, precondition=True, damping=None):
        if precondition:
            pc_src_log = self.precondition(src_log, damping)
        pc_src, src = pc_src_log[1], src_log[1]

        # Compute self-influence scores
        total_influence = 0.0
        for module_name in pc_src.keys():
            pc_src_module = pc_src[module_name]["grad"]
            src_module = src[module_name]["grad"]
            module_influence = reduce(pc_src_module * src_module, "n a b -> n", "sum")
            total_influence += module_influence.squeeze()
        return total_influence

    def compute_influence_all(self, src_log, loader, precondition=True, damping=None):
        if precondition:
            src_log = self.precondition(src_log, damping)

        if_scores = []
        for tgt_log in loader:
            if_scores.append(
                self.compute_influence(src_log, tgt_log, precondition=False)
            )
        return torch.cat(if_scores, dim=-1)

    def get_influence_scores(self):
        return self.influence_scores

    def save_influence_scores(self, filename="influence_scores.csv"):
        self.influence_scores.to_csv(filename, index=True, header=True)
        get_logger().info(f"Influence scores saved to {filename}")
