from __future__ import annotations
import torch
import torch.nn as nn

from monai.networks.nets import DynUNet
from monai.networks.blocks.dynunet_block import UnetResBlock


class ReconDecoder(nn.Module):
    """
    Bottleneck -> image reconstruction decoder (no skip connections).
    Mirrors the number of downsamplings from DynUNet strides.
    """
    def __init__(
        self,
        spatial_dims: int,
        bottleneck_channels: int,
        out_channels: int,
        strides,
        upsample_kernel_size,
        filters,
        res_block: bool = True,
        norm_name=("INSTANCE", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout=None,
    ):
        super().__init__()
        assert spatial_dims == 3, "This decoder assumes 3D."

        n_up = len(strides) - 1  # for your config: 5
        # channel plan: roughly reverse filters
        ch = list(filters[::-1])  # [320,320,256,128,64,32]
        ch = ch[: n_up + 1]       # length 6 for 5 ups

        blocks = []
        in_c = bottleneck_channels
        for i in range(n_up):
            out_c = ch[i + 1]
            k = upsample_kernel_size[::-1][i]  # mirror ordering
            blocks.append(
                nn.ConvTranspose3d(in_c, out_c, kernel_size=k, stride=k, bias=False)
            )
            # refine
            blocks.append(
                UnetResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_c,
                    out_channels=out_c,
                    kernel_size=3,
                    stride=1,
                    norm_name=norm_name,
                    act_name=act_name,
                    dropout=dropout,
                )
            )
            in_c = out_c

        self.net = nn.Sequential(*blocks)
        self.out = nn.Conv3d(in_c, out_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        return self.out(x)


class DynUNetWithRecon(nn.Module):
    """
    DynUNet segmentation + auxiliary reconstruction head from bottleneck.
    Forward returns (seg_logits, recon) by default.
    """
    def __init__(self, recon_out_channels: int = 1, **dynunet_kwargs):
        super().__init__()
        self.seg = DynUNet(**dynunet_kwargs)

        # capture bottleneck via forward hook
        self._bottleneck: torch.Tensor | None = None

        def _hook(_m, _inp, out):
            self._bottleneck = out

        self.seg.bottleneck.register_forward_hook(_hook)

        self.recon = ReconDecoder(
            spatial_dims=dynunet_kwargs["spatial_dims"],
            bottleneck_channels=dynunet_kwargs["filters"][-1],
            out_channels=recon_out_channels,
            strides=dynunet_kwargs["strides"],
            upsample_kernel_size=dynunet_kwargs["upsample_kernel_size"],
            filters=dynunet_kwargs["filters"],
            res_block=dynunet_kwargs.get("res_block", True),
            norm_name=dynunet_kwargs.get("norm_name", ("INSTANCE", {"affine": True})),
        )

    def forward(self, x: torch.Tensor, return_recon: bool = True):
        seg_logits = self.seg(x)
        if not return_recon:
            return seg_logits
        if self._bottleneck is None:
            raise RuntimeError("Bottleneck not captured; hook did not fire.")
        recon = self.recon(self._bottleneck)
        return seg_logits, recon
