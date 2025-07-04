import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, f1_score

from util.misc import MetricLogger


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    clip_grad: Optional[float] = None,
    mixup_fn=None,
    log_writer=None,
    args=None,
) -> dict:
    """Train ``model`` for a single epoch."""
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        optimizer.zero_grad(set_to_none=True)
        loss_scaler(
            loss,
            optimizer,
            clip_grad=clip_grad,
            parameters=model.parameters(),
        )

        metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"])
        if log_writer is not None:
            log_writer.add_scalar("loss/train", loss_value, epoch)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    args,
    epoch: int = 0,
    mode: str = "val",
    num_class: int = 1,
    log_writer=None,
) -> Tuple[dict, float]:
    """Evaluate ``model`` and return metrics and ROC-AUC."""
    criterion = nn.BCEWithLogitsLoss()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test: {mode}" if mode else "Test"

    model.eval()
    all_targets = []
    all_preds = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        metric_logger.update(loss=loss.item())
        all_targets.append(targets.cpu())
        all_preds.append(outputs.sigmoid().cpu())

    metric_logger.synchronize_between_processes()

    if log_writer is not None:
        log_writer.add_scalar(f"loss/{mode}", metric_logger.loss.global_avg, epoch)

    targets_concat = torch.cat(all_targets)
    preds_concat = torch.cat(all_preds)

    # Compute ROC-AUC and F1, handling any potential shape issues gracefully
    try:
        auc = roc_auc_score(targets_concat.numpy(), preds_concat.numpy(), average="macro")
    except Exception:
        auc = 0.0

    try:
        pred_bin = (preds_concat.numpy() >= 0.5).astype(int)
        f1 = f1_score(targets_concat.numpy(), pred_bin, average="macro", zero_division=0)
    except Exception:
        f1 = 0.0

    metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metrics.update({"roc_auc": auc, "f1": f1})
    return metrics, auc, f1
