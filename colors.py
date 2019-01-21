import torch
import torch.nn as nn
import numpy as np
from PIL import Image

def rgb2yuv_(image):
    img = image.transpose(2,0,1).astype('int')    
    Y = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    u = (img[2] - Y) * 0.565
    v = (img[0] - Y) * 0.713
    luv = np.stack([Y, u, v], axis=0)
    luv = np.clip(luv, 0, 255)
    return luv.transpose(1,2,0).astype('uint8').copy()


def rgb2lum_(image):
    img = image.transpose(2,0,1).astype('int')    
    Y = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    lum = Y[None]
    lum = np.clip(lum, 0, 255)
    return lum.transpose(1,2,0).astype('uint8').copy()


def yuv2rgb_(image):
    img = image.astype('float32')
    img = img.transpose(2,0,1)
    R = img[0] + 1.403 * img[2]
    G = img[0] - 0.344 * img[1] - 0.714 * img[2]
    B = img[0] + 1.77 * img[1]
    rgb = np.stack([R, G, B], axis=0)
    rgb = np.clip(rgb, 0, 255)
    return rgb.transpose(1,2,0).astype('uint8').copy()


def transfer_colors(color_src, lum_src):
    content_yuv = rgb2yuv_(np.array(color_src))
    result_lum = rgb2lum_(np.array(lum_src))
    content_yuv[:, :, 0] = result_lum[:, :, 0]
    return Image.fromarray(yuv2rgb_(content_yuv).astype('uint8'), "RGB")


class ImageNetNormalize(nn.Module):
    def __init__(self):
        super(ImageNetNormalize, self).__init__()
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def forward(self, input):
        return (input - self.norm_mean) / self.norm_std


