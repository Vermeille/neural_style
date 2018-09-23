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
            return self.img

        def parameters(self):
            return [self.img]

    def get_parameterized_img(torch_img):
        return SimpleImg(torch_img.clone())


def rgb2yuv(image):
    img = image.transpose(2,0,1).reshape(3,-1)
    luv = np.array([[.299, .587, .114],[-.147, -.288, .436],[.615, -.515, -.1]]).dot(img).reshape((3,image.shape[0],image.shape[1]))
    return luv.transpose(1,2,0)


def rgb2lum(image):
    img = image.transpose(2,0,1).reshape(3,-1)
    luv = np.array([[.299, .587, .114]]).dot(img).reshape((1,image.shape[0],image.shape[1]))
    return luv.transpose(1,2,0)


def yuv2rgb(image):
    img = image.transpose(2,0,1).reshape(3,-1)
    rgb = np.array([[1, 0, 1.139],[1, -.395, -.580],[1, 2.03, 0]]).dot(img).reshape((3,image.shape[0],image.shape[1]))
    return rgb.transpose(1,2,0)


def gram(m):
    b, c, h, w = m.shape
    m = m.view(b, c, h * w)
    m1 = m
    m2 = m.permute(0, 2, 1)
    g = torch.bmm(m1, m2) / (c * h * w)
    return g


def total_variation(input):
    diff_h = ((input[:, :, 1:, :] - input[:, :, :-1, :]) ** 2).mean()
    diff_v = ((input[:, :, :, 1:] - input[:, :, :, :-1]) ** 2).mean()
    return torch.sqrt(diff_h + diff_v)


def normalize(input):
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    return (input - norm_mean) / norm_std


class WithSavedActivations:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.detach = True

        # We want to save activations of convs and relus. Also, MaxPool creates
        # actifacts so we replace them with AvgPool that makes things a bit
        # cleaner.

        for name, layer in self.model.named_children():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(functools.partial(self._save, name))
            if isinstance(layer, nn.ReLU):
                self.model[int(name)] = nn.ReLU(inplace=False)
            if isinstance(layer, nn.MaxPool2d):
                self.model[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)


    def _save(self, name, module, input, output):
        if self.detach:
            self.activations[name] = output.detach().clone()
        else:
            self.activations[name] = output.clone()

    def __call__(self, input, detach):
        self.detach = detach
        self.activations = {}
        self.model(input)
        return self.activations


def npimg_to_tensor(np_img):
    return torch.FloatTensor(np_img / 255).permute(2, 0, 1).cuda()


def artistic_style(content_img, style_img, m=None, style_ratio=1e1, tv_ratio=10):
    content_layers = ['21']
    if m is None:
        m = M.vgg19(pretrained=True).cuda().eval()
        m = WithSavedActivations(m.features)

    torch_img = npimg_to_tensor(content_img)
    photo_activations = m(normalize(torch_img[None]), detach=True)

    torch_style = npimg_to_tensor(style_img)
    style_activations = m(normalize(torch_style[None]), detach=True)

    canvas = torch.randn_like(torch_img, requires_grad=True)

    grams = {layer_id: gram(layer_data).detach()
             for layer_id, layer_data in style_activations.items()}

    canvas = get_parameterized_img(torch_img)
    del torch_img
    del torch_style
    del style_activations

    photo_activations = {
            i: p
            for i, p in photo_activations.items()
            if i in content_layers}

    opt = O.LBFGS(canvas.parameters(), lr=0.5, history_size=10)

    for i in range(50):
        def make_loss():
            gc.collect()
            opt.zero_grad()
            input_img = canvas()
            activations = m(normalize(input_img), detach=False)
            style_loss = 0
            for j in ['0', '5', '10', '19', '28']:
                style_loss += 0.2 * F.mse_loss(
                        gram(activations[j]), grams[j], reduction='sum')

            content_loss = 0
            for j in content_layers:
                content_loss += F.mse_loss(activations[j], photo_activations[j])

            loss = content_loss
            loss += style_ratio * style_loss
            loss += tv_ratio * total_variation(input_img)

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

    result = artistic_style(content, style_img, None, args.ratio, args.tv_ratio)

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
    parser.add_argument('--ratio', default=1e3, type=float)
    parser.add_argument('--tv_ratio', default=10, type=float)
    parser.add_argument('--preserve_colors', default='off')
    args = parser.parse_args(sys.argv[1:])

    result = go(args)
    result.save(args.out)

