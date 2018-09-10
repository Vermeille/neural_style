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

    def __call__(self, input):
        self.activations = {}
        self.model(input)
        return self.activations


def lowres_noise_like(tens):
    lowres_size = (
            tens.shape[0],
            tens.shape[1] // 8,
            tens.shape[2] // 8)

    canvas = torch.randn(*lowres_size) / 50 + 0.5
    canvas = F.interpolate(
            canvas[None],
            size=tens.shape[1:],
            mode='bilinear',
            align_corners=False)[0].detach().cuda()
    canvas.requires_grad = True
    return canvas


def npimg_to_tensor(np_img):
    return torch.FloatTensor(np_img / 255).permute(2, 0, 1).cuda()


def artistic_style(content_img, style_img, m=None, style_ratio=1e1):
    content_layers = ['21']
    if m is None:
        m = M.vgg19(pretrained=True).cuda().eval()
        m = WithSavedActivations(m.features)

    m.detach = True
    torch_img = npimg_to_tensor(content_img)
    photo_activations = m(normalize(torch_img[None]))

    torch_style = npimg_to_tensor(style_img)
    style_activations = m(normalize(torch_style[None]))

    canvas = torch.randn_like(torch_img, requires_grad=True)

    grams = {layer_id: gram(layer_data).detach()
             for layer_id, layer_data in style_activations.items()}

    canvas = torch_img.clone()#lowres_noise_like(torch_img)
    canvas.requires_grad = True
    del torch_img
    del torch_style
    del style_activations

    photo_activations = {
            i: p
            for i, p in photo_activations.items()
            if i in content_layers}

    opt = O.LBFGS([canvas], lr=0.5, history_size=10)

    for i in range(30):
        def make_loss():
            gc.collect()
            opt.zero_grad()
            m.detach = False
            input_img = torch.sigmoid(canvas[None])
            activations = m(normalize(input_img))
            style_loss = 0
            for j in ['0', '5', '10', '19', '28']:
                style_loss += 0.2 * F.mse_loss(
                        gram(activations[j]), grams[j], reduction='sum')

            content_loss = 0
            for j in content_layers:
                content_loss += F.mse_loss(activations[j], photo_activations[j])

            loss = content_loss + style_ratio * style_loss
            loss.backward()
            return loss

        loss = opt.step(make_loss)
        if i % 10 == 0:
            print(i, loss.item())

    return (255 * torch.sigmoid(canvas)
            .cpu().detach().numpy().transpose(1, 2, 0)).astype('uint8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Implementation of Neural Artistic Style by Gatys")
    parser.add_argument('--content', required=True)
    parser.add_argument('--style', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--content_scale', type=float)
    parser.add_argument('--style_scale', type=float)
    args = parser.parse_args(sys.argv[1:])

    result = style_from_uris(args)
    result = Image.fromarray(result)
    result.save(args.out)

