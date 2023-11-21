import torch
from einops import einsum, rearrange, reduce

from analog.analysis import AnalysisBase
from analog.analysis.utils import reconstruct_grad, do_decompose, rescaled_dot_product


class InfluenceFunction(AnalysisBase):
    def parse_config(self):
        return

    @torch.no_grad()
    def precondition(self, src, damping=0.0):
        preconditioned = {}
        if not hasattr(self, "hessian_eigval"):
            (
                self.hessian_eigval,
                self.hessian_eigvec,
            ) = self.hessian_handler.hessian_svd()
        for module_name in src.keys():
            src_log = src[module_name].to("cpu")
            module_eigval = self.hessian_eigval[module_name]
            module_eigvec = self.hessian_eigvec[module_name]
            rotated_grad = einsum(
                module_eigvec["backward"].t(),
                src_log,
                module_eigvec["forward"],
                "a b, batch b c, c d -> batch a d",
            )
            scale = torch.outer(module_eigval["backward"], module_eigval["forward"])
            prec_rotated_grad = rotated_grad / (scale + damping)
            preconditioned[module_name] = einsum(
                module_eigvec["backward"],
                prec_rotated_grad,
                module_eigvec["forward"].t(),
                "a b, batch b c, c d -> batch a d",
            )
        return preconditioned

    @torch.no_grad()
    def compute_influence(self, src, tgt, preconditioned=False, damping=0.0):
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
            total_influence += module_influence.squeeze()
        return total_influence

    def compute_self_influence(self, src, damping=0.0):
        src_pc = self.precondition(src, damping)
        total_influence = 0.0
        for module_name in src_pc.keys():
            src_pc_log, src_log = src_pc[module_name], src[module_name]
            module_influence = reduce(src_pc_log * src_log, "n a b -> n", "sum")
            total_influence += module_influence.squeeze()
        return total_influence

    def compute_influence_all(self, src, loader, damping=0.0):
        if_scores = []
        src = self.precondition(src, damping)
        for tgt_ids, tgt in loader:
            if_scores.append(self.compute_influence(src, tgt, preconditioned=True))
        return torch.cat(if_scores, dim=-1)
