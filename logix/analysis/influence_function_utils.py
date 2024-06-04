# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

import torch
from einops import einsum, rearrange, reduce

from logix.state import LogIXState
from logix.statistic.utils import make_2d
from logix.utils import nested_dict


def precondition_kfac(
    src: Dict[str, Dict[str, torch.Tensor]],
    state: LogIXState,
    damping: Optional[float] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    preconditioned = nested_dict()
    cov_eigval, cov_eigvec = state.get_covariance_svd_state()
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

    return preconditioned


def precondition_raw(
    src: Dict[str, Dict[str, torch.Tensor]],
    state: LogIXState,
    damping: Optional[float] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    preconditioned = nested_dict()
    cov_inverse = state.get_covariance_inverse_state(damping=damping)
    for module_name in src.keys():
        device = src[module_name]["grad"].device
        grad_cov_inverse = cov_inverse[module_name]["grad"].to(device=device)
        original_shape = src[module_name]["grad"].shape
        preconditioned[module_name]["grad"] = (
            make_2d(src[module_name]["grad"], None, "grad") @ grad_cov_inverse
        ).reshape(original_shape)

    return preconditioned


def cross_dot_product(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    assert src.shape[1:] == tgt.shape[1:]
    src_expanded = rearrange(src, "n ... -> n 1 ...")
    tgt_expanded = rearrange(tgt, "m ... -> 1 m ...")
    dot_product_result = reduce(
        src_expanded * tgt_expanded,
        "n m ... -> n m",
        "sum",
    )

    return dot_product_result


def merge_influence_results(
    result_all: Dict[str, Dict[str, torch.Tensor]],
    result: Dict[str, Dict[str, torch.Tensor]],
    axis: str = "tgt",
) -> None:
    assert axis in ["src", "tgt"], f"Unsupported axis {axis}."

    # If merged result is empty, just copy the result and return
    if not result_all:
        result_all.update(result)
        return

    dim = int(axis == "tgt")
    id_key = f"{axis}_ids"

    result_all[id_key].extend(result[id_key])
    if isinstance(result["influence"], dict):
        for key in result_all["influence"].keys():
            result_all["influence"][key] = torch.cat(
                [result_all["influence"][key], result["influence"][key]], dim=dim
            )
    else:
        assert isinstance(result["influence"], torch.Tensor)
        result_all["influence"] = torch.cat(
            [result_all["influence"], result["influence"]], dim=dim
        )
