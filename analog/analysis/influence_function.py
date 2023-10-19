import torch

from analog.analysis import AnalysisBase
from analog.analysis.utils import reconstruct_grad, do_decompose, rescaled_dot_product


class InfluenceFunction(AnalysisBase):
    def parse_config(self):
        return

    @torch.no_grad()
    def compute_influence(self, src, tgt):
        total_influence = 0.0
        for module_name in src.keys():
            hessian_inv = self.hessian_handler.get_hessian_state(module_name)
            src_log, tgt_log = src[module_name], tgt[module_name]
            decompose = do_decompose(src_log, tgt_log)
            module_influence = 1
            if decompose:
                for mode in src_log:
                    module_influence *= rescaled_dot_product(
                        src_log[mode], tgt_log[mode], hessian_inv[mode]
                    )
            else:
                precondition = (
                    hessian_inv["backward"]
                    @ reconstruct_grad(src_log)
                    @ hessian_inv["forward"]
                )
                module_influence = torch.sum(
                    precondition * reconstruct_grad(tgt_log), dim=1
                )
            total_influence += module_influence.squeeze()

        return total_influence

    def compute_self_influence(self, src):
        return self.compute_influence(src, src)

    @torch.no_grad()
    def compute_influence_all(self, src):
        pass
