from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.abstract_task import AbstractTask

BATCH_DTYPE = Dict[str, torch.Tensor]


class TextClassificationModelOutput:
    # Copied from:
    # https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py#L383.
    softmax: nn.Module = torch.nn.Softmax(-1)
    loss_temperature: float = 1.0

    @staticmethod
    def get_output(
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        input_id: torch.Tensor,
        token_type_id: torch.Tensor,
        attention_mask: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        kw_inputs = {
            "input_ids": input_id.unsqueeze(0),
            "token_type_ids": token_type_id.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }

        logits = torch.func.functional_call(
            model, (params, buffers), args=(), kwargs=kw_inputs
        )
        bindex = torch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        logits_correct = logits[bindex, label.unsqueeze(0)]

        cloned_logits = logits.clone()
        cloned_logits[bindex, label.unsqueeze(0)] = torch.tensor(
            -torch.inf, device=logits.device, dtype=logits.dtype
        )

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()

    def get_out_to_loss_grad(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        batch: BATCH_DTYPE,
    ) -> torch.Tensor:
        input_ids, token_type_ids, attention_mask, labels = batch
        kw_inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        logits = torch.func.functional_call(
            model, (params, buffers), args=(), kwargs=kw_inputs
        )
        ps = self.softmax(logits / self.loss_temperature)[
            torch.arange(logits.size(0)), labels
        ]
        return (1 - ps).clone().detach().unsqueeze(-1)


class GlueExperimentTask(AbstractTask):
    def __init__(
        self, device: torch.device = "cpu", generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__(device=device, generator=generator)

    def get_train_loss(
        self,
        model: nn.Module,
        batch: BATCH_DTYPE,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        if parameter_and_buffer_dicts is None:
            inputs = (
                batch["input_ids"].to(self.device),
                batch["token_type_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )
            outputs = model(*inputs)
        else:
            params, buffers = parameter_and_buffer_dicts
            outputs = torch.func.functional_call(
                model,
                (params, buffers),
                args=(
                    batch["input_ids"].unsqueeze(0).to(self.device),
                    batch["token_type_ids"].unsqueeze(0).to(self.device),
                    batch["attention_mask"].unsqueeze(0).to(self.device),
                ),
            )
            batch["labels"] = batch["labels"].unsqueeze(0).to(self.device)

        if not sample:
            return F.cross_entropy(
                outputs, batch["labels"].to(self.device), reduction=reduction
            )
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(outputs, dim=-1)
                sampled_labels = torch.multinomial(
                    probs, num_samples=1, generator=self.generator
                ).flatten()
            return F.cross_entropy(
                outputs, sampled_labels.detach(), reduction=reduction
            )

    def get_measurement(
        self,
        model: nn.Module,
        batch: BATCH_DTYPE,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        assert reduction == "sum"

        if parameter_and_buffer_dicts is None:
            inputs = (
                batch["input_ids"].to(self.device),
                batch["token_type_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )
            logits = model(*inputs)
        else:
            params, buffers = parameter_and_buffer_dicts
            logits = torch.func.functional_call(
                model,
                (params, buffers),
                args=(
                    batch["input_ids"].unsqueeze(0).to(self.device),
                    batch["token_type_ids"].unsqueeze(0).to(self.device),
                    batch["attention_mask"].unsqueeze(0).to(self.device),
                ),
            )
            batch["labels"] = batch["labels"].unsqueeze(0).to(self.device)

        labels = batch["labels"]
        bindex = torch.arange(logits.shape[0]).to(
            device=logits.device, non_blocking=False
        )
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(
            -torch.inf, device=logits.device, dtype=logits.dtype
        )

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()

    def get_batch_size(self, batch: BATCH_DTYPE) -> int:
        if isinstance(batch, list):
            return batch[0].shape[0]
        else:
            return batch["labels"].shape[0]

    def influence_modules(self) -> List[str]:
        total_modules = []

        # Add attention layers:
        for i in range(12):
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.self.query")
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.self.key")
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.self.value")
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.output.dense")

        # Add MLP layers:
        for i in range(12):
            total_modules.append(f"model.bert.encoder.layer.{i}.intermediate.dense")
            total_modules.append(f"model.bert.encoder.layer.{i}.output.dense")

        # Final classification layers:
        total_modules.append("model.bert.pooler.dense")
        total_modules.append("model.classifier")

        return total_modules

    def representation_module(self) -> str:
        return "model.bert.pooler.dense"

    def get_activation_masks(self, batch: Any) -> Optional[torch.Tensor]:
        return batch["attention_mask"].unsqueeze(-1).to(self.device)

    def get_model_output(self) -> Optional[Any]:
        return TextClassificationModelOutput()


class GlueWithLayerNormExperimentTask(GlueExperimentTask):
    def influence_modules(self) -> List[str]:
        total_modules = []

        # Add attention layers:
        for i in range(12):
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.self.query")
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.self.key")
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.self.value")
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.output.dense")

        # Add layer norm layers:
        for i in range(12):
            total_modules.append(
                f"model.bert.encoder.layer.{i}.attention.output.LayerNorm"
            )
            total_modules.append(f"model.bert.encoder.layer.{i}.output.LayerNorm")

        # Add MLP layers:
        for i in range(12):
            total_modules.append(f"model.bert.encoder.layer.{i}.intermediate.dense")
            total_modules.append(f"model.bert.encoder.layer.{i}.output.dense")

        # Final classification layers:
        total_modules.append("model.bert.pooler.dense")
        total_modules.append("model.classifier")

        total_modules.append("model.bert.embeddings.LayerNorm")

        return total_modules


class GlueWithAllExperimentTask(GlueExperimentTask):
    def influence_modules(self) -> List[str]:
        total_modules = []

        # Add attention layers:
        for i in range(12):
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.self.query")
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.self.key")
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.self.value")
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.output.dense")

        # Add layer norm layers:
        for i in range(12):
            total_modules.append(
                f"model.bert.encoder.layer.{i}.attention.output.LayerNorm"
            )
            total_modules.append(f"model.bert.encoder.layer.{i}.output.LayerNorm")

        # Add MLP layers:
        for i in range(12):
            total_modules.append(f"model.bert.encoder.layer.{i}.intermediate.dense")
            total_modules.append(f"model.bert.encoder.layer.{i}.output.dense")

        # Final classification layers:
        total_modules.append("model.bert.pooler.dense")
        total_modules.append("model.classifier")
        total_modules.append("model.bert.embeddings.LayerNorm")
        total_modules.append("model.bert.embeddings.word_embeddings")
        total_modules.append("model.bert.embeddings.position_embeddings")
        total_modules.append("model.bert.embeddings.token_type_embeddings")

        return total_modules


class GlueWithEmbeddingExperimentTask(GlueExperimentTask):
    def influence_modules(self) -> List[str]:
        total_modules = []

        # Add attention layers:
        for i in range(12):
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.self.query")
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.self.key")
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.self.value")
            total_modules.append(f"model.bert.encoder.layer.{i}.attention.output.dense")

        # Add MLP layers:
        for i in range(12):
            total_modules.append(f"model.bert.encoder.layer.{i}.intermediate.dense")
            total_modules.append(f"model.bert.encoder.layer.{i}.output.dense")

        # Final classification layers:
        total_modules.append("model.bert.pooler.dense")
        total_modules.append("model.classifier")
        total_modules.append("model.bert.embeddings.word_embeddings")
        total_modules.append("model.bert.embeddings.position_embeddings")
        total_modules.append("model.bert.embeddings.token_type_embeddings")

        return total_modules
