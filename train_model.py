#!/usr/bin/env python3


import argparse
import os
from config import DEVICE, get_config
from criterions.build import build_criterion
from datasets.build import build_dataset
from models.build import build_model
from monai.metrics import DiceMetric
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from torch import optim as optim
from torch.utils.data import DataLoader
from train import train_model


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

    # dataset
    datasets = build_dataset(cfg=cfg)
    train_dataset = datasets['train']
    val_dataset = datasets['val']

    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.LOADER.TRAIN.BATCH_SIZE,
        shuffle=cfg.LOADER.TRAIN.SHUFFLE,
        num_workers=cfg.LOADER.TRAIN.NUM_WORKERS,
        pin_memory=cfg.LOADER.TRAIN.PIN_MEMORY,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.LOADER.VAL.BATCH_SIZE,
        shuffle=cfg.LOADER.VAL.SHUFFLE,
        num_workers=cfg.LOADER.VAL.NUM_WORKERS,
        pin_memory=cfg.LOADER.VAL.PIN_MEMORY,
    )

    # model
    model = build_model(cfg=cfg)
    print(model)

    # optimizer
    optimizer = build_optimizer(cfg=cfg, model=model)

    # lr scheduler
    lr_scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)

    # criterion
    criterion = build_criterion(cfg=cfg)

    # dice_metric (val)
    dice_metric = DiceMetric(
        include_background=cfg.METRIC.DICE.INCL_BG,
        reduction=cfg.METRIC.DICE.REDUCTION,
        get_not_nans=cfg.METRIC.DICE.GET_NOT_NANS,
        num_classes=cfg.MODEL.N_CLASSES,
    )

    train_model(
        model=model,
        img_size=cfg.DATA.IMG_SIZE,
        ckpt_best=cfg.CKPT.BEST,
        ckpt_last=cfg.CKPT.LAST,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        max_norm=cfg.TRAIN.MAX_NORM,
        dice_metric=dice_metric,
        device=DEVICE,
        n_epochs=cfg.TRAIN.N_EPOCHS,
        val_every_n_epoch=cfg.VAL.VAL_EVERY_N_EPOCH,
        train_loader=train_loader,
        val_loader=val_loader,
        stats_filepath=os.path.join(cfg.CKPT.DIR, cfg.TRAIN.STATS_FILEPATH),
    )

    return


if __name__ == "__main__":
    main()
