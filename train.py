from typing import List, Optional, Tuple, Union

import json
import logger
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from avg_meter import AverageMeter
from logger import log_info, timestamp
from ptflops import get_model_complexity_info
from timm.scheduler.scheduler import Scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from val import val_epoch


def train_epoch(
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: Scheduler,
        criterion: nn.Module,
        max_norm: float,
        device: torch.device,
        epoch: int,
        train_loader: DataLoader,
) -> float:
    model.train(mode=True)

    loader_tqdm = tqdm(iterable=train_loader, position=1)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)

    n_iters = len(train_loader)
    loss_meter = AverageMeter()

    for i, batch in enumerate(iterable=loader_tqdm):
        image = batch["image"].to(device=device)
        label = batch["label"].to(device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = 0.0
        for logit in logits:
            loss += criterion(logit, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(n_iters * epoch + i)

        loss_meter.update(val=loss.item(), n=image.shape[0])
        loader_tqdm.set_description(
            desc=f"[{timestamp()}] [Batch {i+1}] loss {loss_meter.avg:.6f}",
            refresh=True,
        )

    return loss_meter.avg


def train_model(
        model: nn.Module,
        img_size: Union[List[int], Tuple[int]],
        ckpt_best: str,
        ckpt_last: str,
        optimizer: optim.Optimizer,
        lr_scheduler: Optional[Scheduler],
        criterion: nn.Module,
        max_norm: float,
        dice_metric,
        device: torch.device,
        n_epochs: int,
        val_every_n_epoch: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        stats_filepath: str,
) -> None:
    path, _ = os.path.split(p=ckpt_last)
    if not os.path.exists(path=path):
        os.makedirs(name=path, exist_ok=True)

    model.to(device=device)

    macs, params = get_model_complexity_info(
        model=model,
        input_res=tuple(img_size),
        print_per_layer_stat=False,
        as_strings=False,
        verbose=False,
    )
    log_info(f"Macs   {macs * 1e-9:.4f}G")
    log_info(f"Params {params * 1e-6:.4f}M")

    init_epoch = 0
    best_dice = 0.0
    avg_dice_losses = []
    dice_scores = []
    avg_dice_scores = []

    if os.path.exists(path=ckpt_last):
        ckpt = torch.load(f=ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(state_dict=ckpt["lr_scheduler_state_dict"])
        init_epoch = ckpt["epoch"] + 1
        best_dice = ckpt["best_dice"]
        filename = os.path.basename(p=ckpt_last)
        logger.log_info(f"Loaded `{filename}`")

    epoch_tqdm = tqdm(
        iterable=range(init_epoch, n_epochs),
        desc=f"[{timestamp()}] [Epoch {init_epoch}]",
        position=0,
        leave=True,
    )

    for epoch in epoch_tqdm:
        epoch_tqdm.set_description(desc=f"[{timestamp()}] [Epoch {epoch}]", refresh=True)
        for param_group in optimizer.param_groups:
            epoch_tqdm.write(f"[{timestamp()}] [Epoch {epoch}] LR {param_group['lr']}")

        avg_dice_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            max_norm=max_norm,
            device=device,
            epoch=epoch,
            train_loader=train_loader,
        )

        if (epoch + 1) % val_every_n_epoch == 0:
            avg_dice = val_epoch(
                model=model,
                dice_metric=dice_metric,
                device=device,
                val_loader=val_loader,
            )
            dice_scores.append(avg_dice.tolist())
            avg_dice = np.mean(a=avg_dice, dtype=np.float32)

        epoch_tqdm.write(s=f"[{timestamp()}] [Epoch {epoch}] dice loss  {avg_dice_loss:.6f}")
        if (epoch + 1) % val_every_n_epoch == 0:
            epoch_tqdm.write(s=f"[{timestamp()}] [Epoch {epoch}] dice score {avg_dice*100:.4f}")
            avg_dice_scores.append(float(avg_dice))

        avg_dice_losses.append(float(avg_dice_loss))

        if (epoch + 1) % val_every_n_epoch == 0:
            if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save(
                    obj={
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "best_dice": best_dice,
                    },
                    f=ckpt_best,
                )

                epoch_tqdm.write(
                    s=f"[{timestamp()}] [Epoch {epoch}]: Saved best model to "
                    f"'{ckpt_best}'"
                )

        torch.save(
            obj={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
            },
            f=ckpt_last,
        )

    stats = {
        "epoch": list(range(init_epoch, n_epochs)),
        "avg_dice_loss": avg_dice_losses,
        "dice_score": dice_scores,
        "avg_dice_score": avg_dice_scores,
    }

    json_file = open(file=stats_filepath, mode='w')
    json.dump(obj=stats, fp=json_file, indent=4)
    json_file.close()

    return
