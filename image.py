import torch
import torch.nn as nn

class SimpleImg(nn.Module):
    def __init__(self, *shape):
        super(SimpleImg, self).__init__()
        self.img = nn.Parameter(torch.randn(1, *shape) * 0.04)

    def __call__(self):
        return self.img


try:
    import limpid.optvis.param as P
    print('limpid detected')
    class Clamp(nn.Module):
        def __init__(self, canvas):
            super(Clamp, self).__init__()
            self.canvas = canvas

        def forward(self):
            return self.canvas().clamp(-1, 1) / 2 + 0.5

    def get_parameterized_img(*shape, backend='limpid'):
        if backend == 'limpid':
            canvas = P.SpectralImage(shape, 0.01, 0.6)
            #canvas = P.DecorrelatedColors(canvas, sigmoid=False)
            canvas = Clamp(canvas)
            return canvas
        else:
            return Clamp(SimpleImg(*shape))
except:
    def get_parameterized_img(*shape, backend='pixel'):
        assert backend != 'limpid', "Limpid ain't detected"
        return SimpleImg(*shape)

