import torchvision.models as M
import torch.nn as nn
import torch
import functools

class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()

        self.style_layers = ['0', '5', '10', '19', '28']
        self.content_layers = ['21']
        self.cool_layers = self.style_layers + self.content_layers

        self.model = M.vgg19(pretrained=True).features[:max([int(x) for x in self.cool_layers])]
        self.activations = {}
        self.detach = True

        # We want to save activations of convs and relus. Also, MaxPool creates
        # actifacts so we replace them with AvgPool that makes things a bit
        # cleaner.

        for p in self.model.parameters():
            p.requires_grad = False


        for name, layer in self.model.named_children():
            if name in self.cool_layers:
                layer.register_forward_hook(functools.partial(self._save, name))
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
        style_acts = {l: a for l, a in self.activations.items() if l in self.style_layers}
        content_acts = {l: a for l, a in self.activations.items() if l in self.content_layers}
        self.activations = {}
        return style_acts, content_acts

