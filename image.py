import torch
import torch.nn as nn

class SimpleImg(nn.Module):
    def __init__(self, *shape):
        super(SimpleImg, self).__init__()
        self.img = nn.Parameter(torch.randn(1, *shape) * 0.02)

    def __call__(self):
        return torch.sigmoid(self.img - 0.5)


try:
    import limpid.optvis.param as P
    print('limpid detected')

    def get_parameterized_img(*shape, backend='limpid'):
        if backend == 'limpid':
            canvas = P.SpectralImage(shape)
            canvas = P.DecorrelatedColors(canvas, sigmoid=True)
            return canvas
        else:
            return SimpleImg(*shape)
except:
    def get_parameterized_img(*shape, backend='pixel'):
        assert backend != 'limpid', "Limpid ain't detected"
        return SimpleImg(*shape)

