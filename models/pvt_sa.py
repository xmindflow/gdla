from torch import Tensor
from typing import List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.resblk import ResBlk
from modules.upsample import Upsample
from models.pvtv2 import pvt_v2_b2
from modules.decoder.mhsa import MHSA
from registry import MODEL
from timm.layers import trunc_normal_


class PVTSA(nn.Module):
    def __init__(
            self,
            in_chs: int,
            n_classes: int,
            chs: List[int] = [64, 128, 320, 512],
            n_heads: List[int] = [2, 4, 10, 16],
            qkv_bias: bool = True,
            use_rope: bool = False,
            rope_base: float = 10000.0,
            img_size: Tuple[int, int] | None = [224, 224],
            norm_eps: float = 1e-5,
            hidden_ratio: float = 4.0,
            us_ks: int = 3,
            drop_prob: float = 0.,
            deepsupervision: bool = False,
    ) -> None:
        super().__init__()
        self.H, self.W = img_size
        self.deepsupervision = deepsupervision

        self.backbone = pvt_v2_b2()
        path = "pretrained/pvt_v2_b2.pth"
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        print("Loaded pretrained weights")

        # [H/32,W/32] -> [H/16,W/16]
        self.up1 = Upsample(
            in_chs=chs[3],
            out_chs=chs[2],
            ks=us_ks,
        )
        # [H/16,W/16] -> [H/16,W/16]
        self.de1 = MHSA(
            dim=chs[2],
            n_heads=n_heads[2],
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_base=rope_base,
            img_size=[self.H // 16, self.W // 16],
            norm_eps=norm_eps,
            hidden_ratio=hidden_ratio,
            drop_prob=drop_prob,
        )

        # [H/16,W/16] -> [H/8,W/8]
        self.up2 = Upsample(
            in_chs=chs[2],
            out_chs=chs[1],
            ks=us_ks,
        )
        # [H/8,W/8] -> [H/8,W/8]
        self.de2 = MHSA(
            dim=chs[1],
            n_heads=n_heads[1],
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_base=rope_base,
            img_size=[self.H // 8, self.W // 8],
            norm_eps=norm_eps,
            hidden_ratio=hidden_ratio,
            drop_prob=drop_prob,
        )

        # [H/8,W/8] -> [H/4,W/4]
        self.up3 = Upsample(
            in_chs=chs[1],
            out_chs=chs[0],
            ks=us_ks,
        )
        # [H/4,W/4] -> [H/4,W/4]
        self.de3 = MHSA(
            dim=chs[0],
            n_heads=n_heads[0],
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_base=rope_base,
            img_size=[self.H // 4, self.W // 4],
            norm_eps=norm_eps,
            hidden_ratio=hidden_ratio,
            drop_prob=drop_prob,
        )

        self.resblk = ResBlk(
            in_chs=in_chs,
            out_chs=chs[0],
        )

        self.head1 = nn.Conv2d(
            in_channels=chs[0],
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        if self.deepsupervision:
            self.head2 = nn.Conv2d(
                in_channels=chs[1],
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.head3 = nn.Conv2d(
                in_channels=chs[2],
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.head4 = nn.Conv2d(
                in_channels=chs[3],
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        _x = self.resblk(x)
        x = x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)
        x1, x2, x3, x4 = self.backbone(x)

        # decoder
        d3 = self.up1(x4)
        d3 = d3 + x3
        d3 = self.de1(d3)

        d2 = self.up2(d3)
        d2 = d2 + x2
        d2 = self.de2(d2)

        d1 = self.up3(d2)
        d1 = d1 + x1
        d1 = self.de3(d1)

        d1 = F.interpolate(d1, scale_factor=4, mode='bilinear')
        d1 = d1 + _x
        x1 = self.head1(d1)

        if self.deepsupervision:
            x4 = self.head4(x4)
            x4 = F.interpolate(x4, scale_factor=32, mode='bilinear')
            x3 = self.head3(d3)
            x3 = F.interpolate(x3, scale_factor=16, mode='bilinear')
            x2 = self.head2(d2)
            x2 = F.interpolate(x2, scale_factor=8, mode='bilinear')

            return [x4, x3, x2, x1]

        return [x1]


@MODEL.register(name="pvtsa")
def build_model(cfg) -> nn.Module:
    return PVTSA(
        in_chs=cfg.MODEL.IN_CHS,
        n_classes=cfg.MODEL.N_CLASSES,
        chs=cfg.MODEL.CHS,
        n_heads=cfg.MODEL.N_HEADS,
        qkv_bias=cfg.MODEL.QKV_BIAS,
        use_rope=cfg.MODEL.SELF_ATTN.USE_ROPE,
        rope_base=cfg.MODEL.SELF_ATTN.ROPE_BASE,
        img_size=cfg.MODEL.SELF_ATTN.IMG_SIZE,
        norm_eps=cfg.MODEL.NORM_EPS,
        hidden_ratio=cfg.MODEL.HIDDEN_RATIO,
        us_ks=cfg.MODEL.US_KS,
        drop_prob=cfg.MODEL.DROP_PROB,
        deepsupervision=cfg.MODEL.DEEPSUPERVISION,
    )
