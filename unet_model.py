""" Full assembly of the parts to form the complete network """

from unet_part import *
import numpy as np
import torch

class UNet(nn.Module):
    def __init__(self, bilinear=True):
        super(UNet, self).__init__()
        self.bilinear = bilinear

        self.inc = DoubleConv(12, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3)
        self.pixelUnShuffle = nn.PixelUnshuffle(2)
        self.output = nn.Conv2d(3, 12, kernel_size=3, padding=1, stride=1)
        self.pixelShuffle = nn.PixelShuffle(2)

    def forward(self, x):
        in_x = self.pixelUnShuffle(F.interpolate(x, [128, 128], mode='bilinear'))
        x1 = self.inc(in_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        in_x = self.up1(x5, x4)
        in_x = self.up2(in_x, x3)
        in_x = self.up3(in_x, x2)
        in_x = self.up4(in_x, x1)
        x = x + F.interpolate(self.outc(in_x), [x.shape[2], x.shape[3]], mode='bilinear')
        return self.pixelShuffle(self.output(x))[:, :, :, 0:-2]
 
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator


model = UNet()
image = torch.randn(2, 3, 1920, 1080)

sam_checkpoint = "weights/mobile_sam.pt"
model_type = "vit_t"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()
mask_generator = SamAutomaticMaskGenerator(sam)

segtensor = torch.zeros(image.shape)

for k in range(image.shape[0]):

    masks = mask_generator.generate(image[k].squeeze(0))
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    sortedtensor =  torch.zeros((len(sorted_anns), sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
    i = 0
    for ann in sorted_anns:
            m = ann['segmentation']
            sortedtensor[i] = torch.tensor(m).float()
            i = i + 1
    sortedtensor = torch.sum(sortedtensor, dim=0)
    segtensor[k] = sortedtensor / len(sorted_anns)
    
    
    x = torch.cat((image, segtensor), dim=3)
    
    out = model(x)
    
    
    print(out.shape)
