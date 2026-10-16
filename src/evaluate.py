"""Common evaluator for neural LMs: ppl = exp(sum NLL / #non-pad targets) over the whole split."""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def autocast_ctx(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def lm_loss_sum(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> tuple[torch.Tensor, int]:
    """Sum of token NLLs over non-pad targets (loss in fp32) and the number of such targets."""
    nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                          ignore_index=pad_id, reduction="sum")
    return nll, int((targets != pad_id).sum())


@torch.no_grad()
def evaluate_lm(model, inputs: torch.Tensor, targets: torch.Tensor, pad_id: int, device: torch.device,
                amp: bool = True, batch_size: int = 32) -> dict:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for s in range(0, inputs.size(0), batch_size):
        x = inputs[s:s + batch_size].to(device, non_blocking=True)
        y = targets[s:s + batch_size].to(device, non_blocking=True)
        with autocast_ctx(device, amp):
            logits = model(x)
        nll, n = lm_loss_sum(logits, y, pad_id)
        total += float(nll)
        count += n
    if was_training:
        model.train()
    ce = total / count
    return {"nll_sum": total, "count": count, "ce": ce, "ppl": math.exp(ce)}
