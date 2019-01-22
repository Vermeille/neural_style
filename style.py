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
from image import get_parameterized_img
import cv2


def npimg_to_tensor(np_img):
    return transforms.ToTensor()(np_img)


class ArtisticStyleOptimizer:
    def __init__(self, device="cpu"):
        self.loss = StyleLoss().to(device)
        self.device = device


    def build_ref_acts(self, content_img, style_img, style_ratio, content_layers):
        self.loss.set_style(npimg_to_tensor(style_img).to(self.device), style_ratio)
        self.loss.set_content(
                npimg_to_tensor(content_img).to(self.device), content_layers)


    def optimize_img(self, canvas):
        opt = O.LBFGS(canvas.parameters(), lr=0.3, history_size=10)

        prev_loss = None
        for i in range(30):
            def make_loss():
                opt.zero_grad()
                input_img = canvas()
                if True:
                    cv2.imshow('prout',
                            np.array(transforms.ToPILImage()(input_img[0].detach().cpu()))[:, :, ::-1])
                    cv2.waitKey(1)
                loss = self.loss(input_img)
                loss.backward()
                return loss

            loss = opt.step(make_loss).item()
            if prev_loss is not None and loss > prev_loss * 0.95:
                break
            prev_loss = loss
                
        return transforms.ToPILImage()(canvas().cpu().detach()[0])

    def __call__(self, content_img, style_img, style_ratio, content_layers=None):
        self.build_ref_acts(content_img, style_img, style_ratio, content_layers)
        canvas = get_parameterized_img(
                3, content_img.height, content_img.width, backend='limpid').to(self.device)
        return self.optimize_img(canvas)


def go(args, stylizer):
    content = Image.open(args.content)
    content.thumbnail((args.size, args.size))

    style_img = Image.open(args.style)
    if args.scale != 1.0:
        new_style_size = (
                int(style_img.width * args.scale),
                int(style_img.height * args.scale)
        )
        style_img = style_img.resize(new_style_size, Image.BILINEAR)

    result = stylizer(content, style_img, args.ratio, args.content_layers)

    if args.preserve_colors == 'on':
        result = transfer_colors(content, result)

    result.save(args.out)

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
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--content_layers', default=None)
    args = parser.parse_args(sys.argv[1:])

    args.content_layers = args.content_layers and args.content_layers.split(',')
    stylizer = ArtisticStyleOptimizer(device=args.device)
    go(args, stylizer)

