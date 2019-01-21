import torchvision.models as M
import torch.nn as nn
import torch.nn.functional as F
import torch
import functools
from colors import ImageNetNormalize

class PerceptualNet(nn.Module):
    def __init__(self, keep_layers):
        super(PerceptualNet, self).__init__()
        self.keep_layers = keep_layers
        self.model = M.vgg19(pretrained=True).features[:max([int(x) for x in self.keep_layers])]
        self.activations = {}
        self.detach = True

        # We want to save activations of convs and relus. Also, MaxPool creates
        # actifacts so we replace them with AvgPool that makes things a bit
        # cleaner.

        for p in self.model.parameters():
            p.requires_grad = False


        for name, layer in self.model.named_children():
            if name in self.keep_layers:
                layer.register_forward_hook(functools.partial(self._save, name))
            if isinstance(layer, nn.MaxPool2d):
                self.model[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)


    def _save(self, name, module, input, output):
        if self.detach:
            self.activations[name] = output.detach().clone()
        else:
            self.activations[name] = output.clone()

    def forward(self, input, detach):
        self.detach = detach
        self.activations = {}
        self.model(input)
        acts = self.activations
        self.activations = {}
        return acts


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.style_layers = ['0', '5', '10', '19', '28']
        self.content_layers = ['21']
        self.normalize = ImageNetNormalize()
        self.net = PerceptualNet(self.style_layers + self.content_layers)


    def get_style_content_(self, img, detach):
        activations = self.net(self.normalize(img), detach=detach)
        style = {l: a
                for l, a in activations.items()
                if l in self.style_layers}

        content = {l: a
                for l, a in activations.items()
                if l in self.content_layers}

        return style, content


    def set_style(self, style_img, style_ratio):
        self.ratio = style_ratio

        with torch.no_grad():
            activations = self.get_style_content_(style_img[None], detach=True)[0]

        grams = {layer_id: StyleLoss.gram(layer_data)
                 for layer_id, layer_data in activations.items()}

        self.style_grams = grams


    def set_content(self, content_img):
        with torch.no_grad():
            acts = self.get_style_content_(content_img[None], detach=True)[1]
        self.photo_activations = acts


    @staticmethod
    def gram(m):
        b, c, h, w = m.shape
        m = m.view(b, c, h * w)
        m1 = m
        m2 = m.permute(0, 2, 1)
        g = torch.bmm(m1, m2) / (c * h * w)
        return g


    def forward(self, input_img):
        style_acts, content_acts = self.get_style_content_(input_img, detach=False)

        style_loss = 0
        for j in style_acts:
            this_loss = F.mse_loss(
                    StyleLoss.gram(style_acts[j]),
                    self.style_grams[j],
                    reduction='sum')

            style_loss += (1 / len(style_acts)) * this_loss

        content_loss = 0
        for j in content_acts:
            content_loss += F.mse_loss(
                    content_acts[j], self.photo_activations[j])

        return content_loss + self.ratio * style_loss
