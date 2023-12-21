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
    def precondition(self, src, damping=None):
        preconditioned = {}
        (
            hessian_eigval,
            hessian_eigvec,
        ) = self._state.get_hessian_svd_state()
        is_ekfac = hasattr(self._state, "ekfac_eigval_state")
        for module_name in src.keys():
            # if hessian_eigvec is empty, then return src
            if module_name not in hessian_eigvec:
                get_logger().warning(
                    "Hessian has not been computed. No preconditioning applied.\n"
                )
                return src

            src_log = src[module_name].to("cpu")
            module_eigval = hessian_eigval[module_name]
            module_eigvec = hessian_eigvec[module_name]
            rotated_grad = einsum(
                module_eigvec["backward"].t(),
                src_log,
                module_eigvec["forward"],
                "a b, batch b c, c d -> batch a d",
            )
            scale = (
                module_eigval
                if is_ekfac
                else torch.outer(module_eigval["backward"], module_eigval["forward"])
            )
            if damping is None:
                damping = 0.1 * torch.mean(scale)
            prec_rotated_grad = rotated_grad / (scale + damping)
            preconditioned[module_name] = einsum(
                module_eigvec["backward"],
                prec_rotated_grad,
                module_eigvec["forward"].t(),
                "a b, batch b c, c d -> batch a d",
            )
        return preconditioned

    @torch.no_grad()
    def compute_influence(
        self, src, tgt, src_ids=None, tgt_ids=None, preconditioned=False, damping=None
    ):
        if not preconditioned:
            src = self.precondition(src, damping)

        total_influence = 0.0
        for module_name in src.keys():
            src_log, tgt_log = src[module_name], tgt[module_name]
            assert src_log.shape[1:] == tgt_log.shape[1:]
            src_log_expanded = rearrange(src_log, "n ... -> n 1 ...")
            tgt_log_expanded = rearrange(tgt_log, "m ... -> 1 m ...")
            module_influence = reduce(
                src_log_expanded * tgt_log_expanded, "n m a b -> n m", "sum"
            )
            total_influence += module_influence

        if src_ids is not None and tgt_ids is not None:
            assert total_influence.shape[0] == len(src_ids)
            assert total_influence.shape[1] == len(tgt_ids)
            # Ensure src_ids and tgt_ids are in the DataFrame's index and columns, respectively
            self.influence_scores = self.influence_scores.reindex(
                index=src_ids, columns=tgt_ids
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
            ] = total_influence.numpy()

        return total_influence

    def compute_self_influence(self, src_log, damping=None):
        src_ids, src = src_log
        src_pc = self.precondition(src, damping)
        total_influence = 0.0
        for module_name in src_pc.keys():
            src_pc_log, src_log = src_pc[module_name], src[module_name]
            module_influence = reduce(src_pc_log * src_log, "n a b -> n", "sum")
            total_influence += module_influence.squeeze()
        return total_influence

    def compute_influence_all(self, src_log, loader, damping=None):
        if_scores = []
        src_ids, src = src_log
        src = self.precondition(src, damping)
        for tgt_ids, tgt in loader:
            if_scores.append(
                self.compute_influence(
                    src, tgt, src_ids=src_ids, tgt_ids=tgt_ids, preconditioned=True
                )
            )
        return torch.cat(if_scores, dim=-1)

    def get_influence_scores(self):
        return self.influence_scores

    def save_influence_scores(self, filename):
        self.influence_scores.to_csv(filename, index=True, header=True)
        get_logger().info(f"Influence scores saved to {filename}")
