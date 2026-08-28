import torch.nn as nn


class Upsample(nn.Module):
    def __init__(self, in_chs: int, out_chs: int, ks: int = 3) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels=in_chs,
            out_channels=out_chs,
            kernel_size=ks,
            stride=2,
            padding=ks // 2,
            output_padding=1,
            bias=False,
        )

    def forward(self, x):
        x = self.up(x)

        return x
