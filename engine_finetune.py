import math
import sys
import torch
import numpy as np
from typing import Iterable, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
    average_precision_score,
    cohen_kappa_score,
)
from timm.utils import ModelEma


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn = None,
                    set_training_mode=True,
                    log_writer=None,
                    args=None):
    model.train(set_training_mode)
    metric_logger = {}
    for data_iter_step, (samples, targets) in enumerate(data_loader):
        samples, targets = samples.to(device), targets.to(device)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        if log_writer is not None:
            log_writer.update(loss=loss_value)
            log_writer.set_step()

    return {"loss": loss_value}


def evaluate(data_loader, model, device, args, epoch=0, mode="val", log_writer=None):
    model.eval()
    all_true_labels = []
    all_pred_logits = []

    for batch in data_loader:
        images, targets = batch
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)

        all_true_labels.append(targets.cpu().numpy())
        all_pred_logits.append(outputs.cpu().numpy())

    true_labels = np.concatenate(all_true_labels, axis=0)
    pred_logits = np.concatenate(all_pred_logits, axis=0)

    if args.num_classes == 1:
        pred_labels = (pred_logits > 0.5).astype(int)
    else:
        pred_labels = (pred_logits > 0.5).astype(int)

    true_labels = true_labels.astype(int)

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    roc_auc = roc_auc_score(true_labels, pred_logits, average='macro')
    hamming = hamming_loss(true_labels, pred_labels)
    jaccard = jaccard_score(true_labels, pred_labels, average='macro')
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    average_precision = average_precision_score(true_labels, pred_logits, average='macro')
    kappa = cohen_kappa_score(true_labels.argmax(axis=1), pred_labels.argmax(axis=1))

    stats = {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "hamming_loss": hamming,
        "jaccard_score": jaccard,
        "precision": precision,
        "recall": recall,
        "average_precision": average_precision,
        "kappa": kappa,
    }

    return stats, f1
