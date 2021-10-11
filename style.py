import readline
import sys
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.optim as O
import argparse
from torchelie.recipes.neural_style import NeuralStyle
from colors import transfer_colors
import torch


def go(args, stylizer):
    content = Image.open(args.content).convert('RGB')
    if args.size != -1:
        content.thumbnail((args.size, args.size))

    style_img = Image.open(args.style).convert('RGB')
    if args.scale != 1.0:
        new_style_size = (int(style_img.width * args.scale),
                          int(style_img.height * args.scale))
        style_img = style_img.resize(new_style_size, Image.BICUBIC)

    result = stylizer.fit(2000,
                          content,
                          style_img,
                          args.ratio,
                          second_scale_ratio=args.coarse,
                          content_layers=args.content_layers,
                          init_with_content=args.init_with_content)

    if args.preserve_colors:
        result = transfer_colors(content, result)

    transforms.functional.to_pil_image(result).save(args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Implementation of Neural Artistic Style by Gatys")
    parser.add_argument('--content', required=True)
    parser.add_argument('--style', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', type=int)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--ratio', default=1, type=float)
    parser.add_argument('--coarse', default=1, type=float)
    parser.add_argument('--preserve_colors', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--content_layers',
                        default=None,
                        type=lambda x: x and x.split(','))
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args(sys.argv[1:])

    stylizer = NeuralStyle(device=args.device)
    go(args, stylizer)
    if args.interactive:
        while True:
            try:
                cmd = input('> ')
            except:
                break
            try:
                go(parser.parse_args(cmd.split()), stylizer)
            except KeyboardInterrupt:
                pass
            except Exception as e:
                print(str(e))
                pass
