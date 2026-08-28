import logger
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from avg_meter import AverageMeter
from logger import timestamp
from medpy import metric
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from tqdm import tqdm


def val_epoch(
        model: nn.Module,
        dice_metric,
        device: torch.device,
        val_loader: DataLoader,
        batch_size: int = 32,
) -> np.ndarray:
    model.eval()

    loader_tqdm = tqdm(iterable=val_loader, position=1)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)

    dice_meter = AverageMeter()
    acc_meter = AverageMeter()
    hd95_meter = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(iterable=loader_tqdm):
            image = batch["image"].to(device=device)
            label = batch["label"].to(device=device)

            if image.ndim == 4:
                logits = model(image)
                if isinstance(logits, (list, tuple)):
                    logits = logits[-1]
                probs = F.softmax(input=logits, dim=1)
                preds = torch.argmax(input=probs, dim=1, keepdim=True)
                acc = (preds == label.unsqueeze(dim=1)).float().mean()
                acc_meter.update(val=acc.cpu().numpy(), n=label.shape[0])
            elif image.ndim == 5:
                assert image.size(dim=0) == 1, "Volumetric inference assumes batch size = 1"

                D = image.size(dim=2)
                image = image.squeeze(dim=0).permute(1, 0, 2, 3)

                preds = []
                for start in range(0, D, batch_size):
                    end = min(start + batch_size, D)
                    batch = image[start:end]
                    logits = model(batch)
                    if isinstance(logits, (list, tuple)):
                        logits = logits[-1]
                    probs = F.softmax(input=logits, dim=1)
                    pred = torch.argmax(input=probs, dim=1, keepdim=True)  # not sure what dtype

                    if pred.shape[-2] != label.shape[-2] or pred.shape[-1] != label.shape[-1]:
                        pred = pred.cpu().detach().numpy()
                        pred = zoom(
                            input=pred,
                            zoom=(1.0, 1.0, label.shape[-2] / pred.shape[-2], label.shape[-1] / pred.shape[-1]),
                            order=0,
                        )
                        pred = torch.from_numpy(pred).to(device=label.device)

                    preds.append(pred)
                preds = torch.cat(tensors=preds, dim=0)
                preds = preds.permute(1, 0, 2, 3).unsqueeze(dim=0)

                # HD95 is computed per class over the whole volume. It is
                # commented out because it is very slow during evaluation
                # (e.g. Synapse loops over all organ classes); uncomment it
                # to report the 95th-percentile Hausdorff distance.
                # hd95 = []
                # for i in range(1, logits.size(dim=1)):
                #     result = (preds.squeeze(dim=0).squeeze(dim=0).cpu().numpy() == i).astype(np.uint8)
                #     result[result > 0] = 1
                #     reference = (label.squeeze(dim=0).cpu().numpy() == i).astype(np.uint8)
                #     reference[reference > 0] = 1
                #     if result.sum() > 0 and reference.sum() > 0:
                #         hd95.append(metric.binary.hd95(result, reference))
                #     else:
                #         hd95.append(0.0)
                # hd95_meter.update(val=np.asarray(hd95), n=1)
            else:
                raise ValueError(
                    f"Unsupported input tensor shape {image.shape}. "
                    f"Expected 4D [B, C, H, W] or 5D [B, C, D, H, W] for 2D/3D data."
                )

            dice_metric.reset()
            dice_metric(y_pred=preds, y=label.unsqueeze(dim=1))
            dices, not_nans = dice_metric.aggregate()
            dice_meter.update(val=dices.cpu().numpy(), n=not_nans.cpu().numpy())

            avg_dice = np.mean(a=dice_meter.avg)
            avg_hd95 = np.mean(a=hd95_meter.avg)
            loader_tqdm.set_description(
                desc=f"[{timestamp()}] [Batch {i+1}] dice {avg_dice*100:.4f} "
                     f"hd95 {avg_hd95:.4f} "
                     f"acc {acc_meter.avg*100:.4f}",
                refresh=True,
            )

    return dice_meter.avg


def val_model(
        model: nn.Module,
        ckpt_filepath: str,
        dice_metric,
        device: torch.device,
        dataloader: DataLoader,
) -> np.ndarray:
    model.to(device=device)

    if os.path.exists(path=ckpt_filepath):
        ckpt = torch.load(f=ckpt_filepath, map_location=device, weights_only=False)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        filename = os.path.basename(ckpt_filepath)
        logger.log_info(f"Loaded `{filename}`.")
    else:
        logger.log_error(f"Cannot find ckpt file at `{ckpt_filepath}`")
        exit(1)

    dice_score = val_epoch(
        model=model,
        dice_metric=dice_metric,
        device=device,
        val_loader=dataloader,
    )

    return dice_score
