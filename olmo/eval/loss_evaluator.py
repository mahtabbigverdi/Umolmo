"""Class to build metrics for a model based on the loss"""
import logging
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, Optional, Union, List

import torch
from torch.utils.data import DataLoader
from torchmetrics import Metric, MeanMetric
from tqdm import tqdm

from olmo.torch_util import move_to_device

__all__ = ["LossDatasetEvaluator", "LossMetrics"]

log = logging.getLogger(__name__)


class LossMetrics:

    def __init__(self, device, collect_outputs=False):
        self.eval_metrics: Dict[str, MeanMetric] = dict(
            CrossEntropyLoss=MeanMetric("error").to(device),
            ZLoss=MeanMetric("error").to(device),
            Accuracy=MeanMetric("error").to(device),
        )

    def reset(self) -> None:
        if isinstance(self.eval_metrics, Metric):
            self.eval_metrics.reset()
        else:
            for metric in self.eval_metrics.values():
                metric.reset()

    def compute(self) -> Dict[str, float]:
        return {k: v.compute().item()
                for k, v in self.eval_metrics.items() if v.weight > 0}

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        model_out,
        cross_entropy_loss: torch.Tensor,
        zloss: torch.Tensor
    ) -> None:
        loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
        total_weight = loss_masks.sum()
        labels = batch["labels"]
        pred = torch.argmax(model_out.logits, dim=-1)
        accuracy = (pred.flatten() == labels.flatten()).float().sum().item()
        self.eval_metrics["CrossEntropyLoss"].update(cross_entropy_loss/total_weight, total_weight)
        if zloss is not None:
            self.eval_metrics["ZLoss"].update(zloss/total_weight, total_weight)
        self.eval_metrics["Accuracy"].update(accuracy/total_weight, total_weight)
        if model_out.metrics is not None:
            for name, val in model_out.metrics.items():
                if name not in self.eval_metrics:
                    self.eval_metrics[name] = MeanMetric("error").to(cross_entropy_loss.device)
                if isinstance(val, tuple):
                    self.eval_metrics[name].update(val[0]/val[1], val[1])
                else:
                    self.eval_metrics[name].update(val, 1)


@dataclass
class LossDatasetEvaluator:
    """Evaluates a model on a dataset based its on its loss and other forward-pass metrics"""
    label: str
    eval_loader: DataLoader
    evaluator: LossMetrics
    num_batches: Optional[int] = None
    z_loss: Optional[float] = None
    console_log_interval: Optional[int] = None

    def run(self, model, device, autocast_precision, loss_fn=None, pbar=False):
        # Reset metrics.
        self.evaluator.reset()
        if loss_fn is None:
            # FIXME not a default arg to avoid circular imports
            from olmo.train import cross_entropy_loss as loss_fn

        # Initialize data loader iterator.
        eval_batches = iter(self.eval_loader)

        # Adjust how many batches to evaluate on.
        num_eval_batches = self.num_batches
        if num_eval_batches > 0:
            if isinstance(self.eval_loader, torch.utils.data.IterableDataset):
                # No defined length
                num_eval_batches = None
            else:
                num_eval_batches = min(num_eval_batches, len(self.eval_loader))
            eval_batches = islice(eval_batches, num_eval_batches)

        # Run model over batches.
        with torch.inference_mode():
            for eval_step, batch in enumerate(tqdm(eval_batches, total=num_eval_batches, disable=not pbar)):
                batch = move_to_device(batch, device)
                response_mask = (batch["loss_masks"] > 0)
                with torch.autocast("cuda", enabled=True, dtype=autocast_precision):
                    model_out = model(
                        **{k: v for k,v in batch.items() if k not in ["labels", "loss_masks", "metadata"]},
                        response_mask=response_mask)
                logits = model_out.logits
                loss_masks = batch["loss_masks"]
                loss_masks = loss_masks * (loss_masks > 0)
                labels = batch["labels"].long()
                labels.masked_fill_(~(loss_masks > 0), -100)
                labels = labels.view(-1)
                logits_for_loss = logits.to(torch.float32).view(-1, logits.size(-1)) # for numerical stability
                ce_loss, z_loss = loss_fn(
                    logits_for_loss, labels, ignore_index=-100, reduction="sum",
                    compute_z_loss=self.z_loss is not None, z_loss_scale=self.z_loss,
                )
                self.evaluator.update(batch, model_out, ce_loss, z_loss)
            # Log to console.
            if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.console_log_interval == 0:
                log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")

        if hasattr(self.eval_loader, "reset"):
            self.eval_loader.reset()  # Reset the loader to free RAM
        return self.evaluator.compute()
