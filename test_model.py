#!/usr/bin/env python3


import argparse
from config import DEVICE, get_config
import numpy as np
from logger import log_info
from datasets.build import build_dataset
from models.build import build_model
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader
from val import val_model


def main() -> None:
    parser = argparse.ArgumentParser(prog='2D Medical Image Segementation')
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        metavar="FILE",
        help='path to config file',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        metavar="FILE",
        help='path to dataset config file',
    )
    parser.add_argument(
        '--opts',
        nargs=argparse.REMAINDER,
        default=None,
        help="override config options, e.g. --opts CKPT.BEST /path/to/best.ckpt",
    )
    args, unparsed = parser.parse_known_args()
    cfg = get_config(args=args)

    dataset = build_dataset(cfg=cfg)['test']

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.LOADER.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.LOADER.TEST.NUM_WORKERS,
        pin_memory=cfg.LOADER.TEST.PIN_MEMORY,
    )

    model = build_model(cfg=cfg)
    print(model)

    # acc fn (val)
    dice_metric = DiceMetric(
        include_background=cfg.METRIC.DICE.INCL_BG,
        reduction=cfg.METRIC.DICE.REDUCTION,
        get_not_nans=cfg.METRIC.DICE.GET_NOT_NANS,
        num_classes=cfg.MODEL.N_CLASSES,
    )

    dice = val_model(
        model=model,
        ckpt_filepath=cfg.CKPT.BEST,
        dice_metric=dice_metric,
        device=DEVICE,
        dataloader=dataloader,
    )
    print(dice)

    avg_dice_score = np.mean(a=dice, dtype=np.float32)

    log_info(f"Dice Score {dice}")
    log_info(f"Avg. Dice Score: {avg_dice_score.item():.6f}")

    return


if __name__ == "__main__":
    main()
