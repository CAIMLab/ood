import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, channels_img, channels_latent, features_e):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_e, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_e, features_e * 2, 4, 2, 1),
            self._block(features_e * 2, features_e * 4, 4, 2, 1),
            self._block(features_e * 4, features_e * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_e * 8, channels_latent, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.enc(x)