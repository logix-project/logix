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
#
from typing import Dict, List, Optional, Tuple

import torch
from einops import reduce
from tqdm import tqdm

from logix.analysis.influence_function_utils import (
    cross_dot_product,
    merge_influence_results,
    precondition_kfac,
    precondition_raw,
)
from logix.state import LogIXState
from logix.utils import flatten_log, get_logger, synchronize_device, unflatten_log


class InfluenceFunction:
    def __init__(self, state: LogIXState):
        # state
        self._state = state

        self.influence_scores = {}
        self.self_influence_scores = {}

    def get_influence_scores(self):
        return self.influence_scores

    def get_self_influence_scores(self):
        return self.self_influence_scores

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
        assert hessian in ["auto", "kfac", "raw"], f"Invalid hessian {hessian}"

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
        hessian: Optional[str] = "auto",
        influence_groups: Optional[List[str]] = None,
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
                src_log=src_log, damping=damping, hessian=hessian
            )

        src_ids, src = src_log
        tgt_ids, tgt = tgt_log

        # Initialize influence scores
        total_influence = {"total": 0}
        for influence_group in influence_groups or []:
            total_influence[influence_group] = 0

        # Compute influence scores. By default, we should compute the basic influence
        # scores, which is essentially the inner product between the source and target
        # gradients. If mode is cosine, we should normalize the influence score by the
        # L2 norm of the target gardients. If mode is l2, we should subtract the L2
        # norm of the target gradients.
        if not isinstance(tgt, dict):
            assert isinstance(tgt, torch.Tensor)
            src = flatten_log(
                log=src, path=self._state.get_state("model_module")["path"]
            )
            tgt = tgt.to(device=src.device)
            total_influence["total"] += cross_dot_product(src, tgt)
        else:
            synchronize_device(src, tgt)
            for module_name in src.keys():
                module_influence = cross_dot_product(
                    src[module_name]["grad"], tgt[module_name]["grad"]
                )
                total_influence["total"] += module_influence
                if influence_groups is not None:
                    groups = [g for g in influence_groups if g in module_name]
                    for group in groups:
                        total_influence[group] += module_influence

        if mode == "cosine":
            tgt_norm = self.compute_self_influence(
                tgt_log,
                precondition=True,
                hessian=hessian,
                influence_groups=influence_groups,
                damping=damping,
            ).pop("influence")
            for key in total_influence.keys():
                tgt_norm_key = tgt_norm if influence_groups is None else tgt_norm[key]
                total_influence[key] /= torch.sqrt(tgt_norm_key.unsqueeze(0))
        elif mode == "l2":
            tgt_norm = self.compute_self_influence(
                tgt_log,
                precondition=True,
                hessian=hessian,
                influence_groups=influence_groups,
                damping=damping,
            ).pop("influence")
            for key in total_influence.keys():
                tgt_norm_key = tgt_norm if influence_groups is None else tgt_norm[key]
                total_influence[key] -= 0.5 * tgt_norm_key.unsqueeze(0)

        # Move influence scores to CPU to save memory
        for key, value in total_influence.items():
            assert value.shape[0] == len(src_ids)
            assert value.shape[1] == len(tgt_ids)
            total_influence[key] = value.cpu()

        result["src_ids"] = list(src_ids)
        result["tgt_ids"] = list(tgt_ids)
        result["influence"] = (
            total_influence.pop("total")
            if influence_groups is None
            else total_influence
        )

        return result

    @torch.no_grad()
    def compute_self_influence(
        self,
        src_log: Tuple[str, Dict[str, Dict[str, torch.Tensor]]],
        precondition: Optional[bool] = True,
        hessian: Optional[str] = "auto",
        influence_groups: Optional[List[str]] = None,
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

        tgt = src
        if precondition:
            tgt = self.precondition(src_log, hessian=hessian, damping=damping)[1]

        # Initialize influence scores
        total_influence = {"total": 0}
        for influence_group in influence_groups or []:
            total_influence[influence_group] = 0

        # Compute self-influence scores
        for module_name in src.keys():
            src_module = src[module_name]["grad"]
            tgt_module = tgt[module_name]["grad"] if tgt is not None else src_module
            module_influence = reduce(
                src_module * tgt_module, "n a b -> n", "sum"
            ).reshape(-1)
            total_influence["total"] += module_influence
            if influence_groups is not None:
                groups = [g for g in influence_groups if g in module_name]
                for group in groups:
                    total_influence[group] += module_influence

        # Move influence scores to CPU to save memory
        for key, value in total_influence.items():
            assert len(value) == len(src_ids)
            total_influence[key] = value.cpu()

        result["src_ids"] = src_ids
        result["influence"] = (
            total_influence.pop("total")
            if influence_groups is None
            else total_influence
        )

        return result

    def compute_influence_all(
        self,
        src_log: Tuple[str, Dict[str, Dict[str, torch.Tensor]]],
        loader: torch.utils.data.DataLoader,
        mode: Optional[str] = "dot",
        precondition: Optional[bool] = True,
        save: Optional[bool] = False,
        hessian: Optional[str] = "auto",
        influence_groups: Optional[List[str]] = None,
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
            src_log = self.precondition(src_log, hessian=hessian, damping=damping)

        result_all = {}
        for tgt_log in tqdm(loader, desc="Compute IF"):
            result = self.compute_influence(
                src_log=src_log,
                tgt_log=tgt_log,
                mode=mode,
                precondition=False,
                hessian=hessian,
                influence_groups=influence_groups,
                damping=damping,
            )
            merge_influence_results(result_all, result, axis="tgt")

        if save:
            merge_influence_results(self.influence_scores, result_all, axis="src")

        return result_all
