import torchvision.models as M
import torch.nn as nn
import torch
import functools

class PerceptualNet(nn.Module):
    layer_names = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
                'conv3_4', 'relu3_4', 'maxpool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
                'conv4_4', 'relu4_4', 'maxpool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
                'conv5_4', 'relu5_4', 'maxpool5'
    ]

    def __init__(self, keep_layers):
        super(PerceptualNet, self).__init__()
        self.model = M.vgg19(pretrained=True).features
        self.activations = {}
        self.hooks = []
        self.detach = True

        # We want to save activations of convs and relus. Also, MaxPool creates
        # actifacts so we replace them with AvgPool that makes things a bit
        # cleaner.

        for p in self.model.parameters():
            p.requires_grad = False


        for name, layer in self.model.named_children():
            idx = int(name)
            if isinstance(layer, nn.MaxPool2d):
                self.model[idx] = nn.AvgPool2d(kernel_size=2, stride=2)

        self.set_keep_layers(keep_layers)


    def set_keep_layers(self, keep_layers):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        for name, layer in self.model.named_children():
            idx = int(name)
            pretty = PerceptualNet.layer_names[idx]
            if pretty in keep_layers:
                layer.register_forward_hook(functools.partial(self._save, pretty))


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


