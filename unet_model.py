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
 
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
    #img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        #print(m.shape)
        img = m
        #color_mask = np.concatenate([np.random.random(3), [0.35]])
        #img[m] = color_mask
    return img

import sys
sys.path.append("..")
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



if __name__ == '__main__':
    
    sam_checkpoint = r"D:\Users\Downloads\MobileSAM-master\weights/mobile_sam.pt"
    model_type = "vit_t"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)
    
    
    
    from torchsummary import summary
    model = UNet()
    x = torch.randn(1, 3, 1920, 1080)
    
    sorted_x = torch.zeros((x.shape[0], x.shape[1], x.shape[2]))
    
    
    for i in range(sorted_x.shape[0]):
        masks = mask_generator.generate(x[0])
        feature = show_anns(masks)
        #print(feature.shape)
        sorted_x[i] = torch.from_numpy(feature)
    
    #print(sorted_x.shape)
    
    x = torch.cat((x, sorted_x.unsqueeze(3)), dim=3)
    
    out = model(x)
    
    
    print(out.shape)
    
    #print(out.shape)    # torch.Size([1, 3, 256, 256])
    
    summary(model, (3, 1080, 1921), device='cpu')   
