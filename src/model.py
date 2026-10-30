import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet2D(nn.Module):
    """Compact U-Net baseline for pixel-wise seismic fault segmentation."""
    def __init__(self, in_channels=1, out_channels=1, base=16):
        super().__init__()
        self.enc1=ConvBlock(in_channels,base)
        self.enc2=ConvBlock(base,base*2)
        self.enc3=ConvBlock(base*2,base*4)
        self.pool=nn.MaxPool2d(2)
        self.up2=nn.ConvTranspose2d(base*4,base*2,2,2)
        self.dec2=ConvBlock(base*4,base*2)
        self.up1=nn.ConvTranspose2d(base*2,base,2,2)
        self.dec1=ConvBlock(base*2,base)
        self.out=nn.Conv2d(base,out_channels,1)

    def forward(self,x):
        a=self.enc1(x)
        b=self.enc2(self.pool(a))
        c=self.enc3(self.pool(b))
        x=self.up2(c)
        x=nn.functional.interpolate(x,size=b.shape[-2:],mode="bilinear",align_corners=False) if x.shape[-2:]!=b.shape[-2:] else x
        x=self.dec2(torch.cat([x,b],1))
        x=self.up1(x)
        x=nn.functional.interpolate(x,size=a.shape[-2:],mode="bilinear",align_corners=False) if x.shape[-2:]!=a.shape[-2:] else x
        return self.out(self.dec1(torch.cat([x,a],1)))
