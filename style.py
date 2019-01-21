import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from torchvision import models as M
from torchvision import transforms
import functools
from PIL import Image
import requests
import torch.optim as O
import io
import argparse
from perceptual import PerceptualNet
from colors import rgb2yuv, rgb2yuv, yuv2rgb, ImageNetNormalize

try:
    import limpid.optvis.param as P
    print('limpid detected')

    def get_parameterized_img(torch_img):
        canvas = P.SpectralImage((3, torch_img.shape[1], torch_img.shape[2]))
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

    def get_parameterized_img(torch_img):
        return SimpleImg(torch_img.clone())



def gram(m):
    b, c, h, w = m.shape
    m = m.view(b, c, h * w)
    m1 = m
    m2 = m.permute(0, 2, 1)
    g = torch.bmm(m1, m2) / (c * h * w)
    return g


def npimg_to_tensor(np_img):
    return transforms.ToTensor()(np_img)


def artistic_style(content_img, style_img, m=None, style_ratio=1e1):
    normalize = ImageNetNormalize()
    m = PerceptualNet()

    torch_img = npimg_to_tensor(content_img)
    photo_activations = m(normalize(torch_img[None]), detach=True)[1]

    torch_style = npimg_to_tensor(style_img)
    style_activations = m(normalize(torch_style[None]), detach=True)[0]

    canvas = torch.randn_like(torch_img, requires_grad=True)

    grams = {layer_id: gram(layer_data).detach()
             for layer_id, layer_data in style_activations.items()}

    canvas = get_parameterized_img(torch_img)
    del torch_img
    del torch_style
    del style_activations

    opt = O.LBFGS(canvas.parameters(), lr=0.5, history_size=10)

    for i in range(50):
        def make_loss():
            gc.collect()
            opt.zero_grad()
            input_img = canvas()
            style_acts, content_acts = m(normalize(input_img), detach=False)
            style_loss = 0
            for j in style_acts:
                style_loss += (1/len(style_acts)) * F.mse_loss(
                        gram(style_acts[j]), grams[j], reduction='sum')

            content_loss = 0
            for j in content_acts:
                content_loss += F.mse_loss(content_acts[j], photo_activations[j])

            loss = content_loss
            loss += style_ratio * style_loss

            loss.backward()
            return loss

        loss = opt.step(make_loss)
        if i % 10 == 0:
            print(i, loss.item())

    return (255 * canvas()[0]
            .cpu().detach().numpy().transpose(1, 2, 0)).astype('uint8')


def open_img(path, size=None):
    img = Image.open(path)
    if size is not None:
        img.thumbnail((size, size))
    return np.array(img)


def go(args):
    content_scale = args.size
    style_scale = content_scale

    content = open_img(args.content, content_scale)
    style_img = open_img(args.style, style_scale)

    result = artistic_style(content, style_img, None, args.ratio)

    if args.preserve_colors == 'on':
        content_yuv = rgb2yuv(content)
        result_lum = rgb2lum(result)
        content_yuv[:, :, 0] = result_lum[:, :, 0]
        result = yuv2rgb(content_yuv).astype('uint8')

    return Image.fromarray(result, 'RGB')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Implementation of Neural Artistic Style by Gatys")
    parser.add_argument('--content', required=True)
    parser.add_argument('--style', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', type=int)
    parser.add_argument('--ratio', default=1, type=float)
    parser.add_argument('--preserve_colors', default='off')
    args = parser.parse_args(sys.argv[1:])

    result = go(args)
    result.save(args.out)

