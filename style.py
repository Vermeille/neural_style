import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as M
from torchvision import transforms
import functools
from PIL import Image
import requests
import torch.optim as O
import io
import argparse
from perceptual import StyleLoss
from colors import transfer_colors

try:
    import limpid.optvis.param as P
    print('limpid detected')

    def get_parameterized_img(*shape):
        canvas = P.SpectralImage(shape)
        canvas = P.DecorrelatedColors(canvas, sigmoid=True).cuda()
        return canvas
except:
    class SimpleImg:
        def __init__(self, img):
            self.img = img[None]
            self.img.requires_grad = True

        def __call__(self):
            return torch.sigmoid(self.img - 0.5)

        def parameters(self):
            return [self.img]

    def get_parameterized_img(*shape):
        return SimpleImg(torch.randn(*shape))



def npimg_to_tensor(np_img):
    return transforms.ToTensor()(np_img)


class ArtisticStyle:
    def __init__(self, device="cpu"):
        self.loss = StyleLoss().to(device)


    def build_ref_acts(self, content_img, style_img, style_ratio):
        self.loss.set_style(npimg_to_tensor(style_img), style_ratio)
        self.loss.set_content(npimg_to_tensor(content_img))


    def optimize_img(self, canvas):
        #opt = O.LBFGS(canvas.parameters(), lr=0.5, history_size=10)
        opt = O.Adam(canvas.parameters(), lr=0.5)

        for i in range(100):
            def make_loss():
                opt.zero_grad()
                input_img = canvas()
                loss = self.loss(input_img)
                loss.backward()
                return loss

            loss = opt.step(make_loss)
            if i % 10 == 0:
                print(i, loss.item())

        return transforms.ToPILImage()(canvas()[0])

    def __call__(self, content_img, style_img, style_ratio):
        self.build_ref_acts(content_img, style_img, style_ratio)
        canvas = get_parameterized_img(3, content_img.height, content_img.width)
        return self.optimize_img(canvas)


def go(args):
    style_scale = args.scale

    content = Image.open(args.content)
    content.thumbnail((args.size, args.size))

    style_img = Image.open(args.style)
    if style_scale != 1.0:
        new_style_size = (
                style_img.width * style_scale,
                style_img.height * style_scale
        )
        style_img = style_img.resize(new_style_size, Image.BILINEAR)

    stylizer = ArtisticStyle()
    result = stylizer(content, style_img, args.ratio)

    if args.preserve_colors == 'on':
        result = transfer_colors(content, result)

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Implementation of Neural Artistic Style by Gatys")
    parser.add_argument('--content', required=True)
    parser.add_argument('--style', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', type=int)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--ratio', default=1, type=float)
    parser.add_argument('--preserve_colors', default='off')
    args = parser.parse_args(sys.argv[1:])

    result = go(args)
    result.save(args.out)

