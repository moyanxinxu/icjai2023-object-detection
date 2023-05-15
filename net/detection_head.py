from torch import nn

def confs(num_classes):
    layers = []
    layers += [nn.Conv2d(512, num_classes * 4, 3, padding=1)]
    layers += [nn.Conv2d(1024, num_classes * 6, 3, padding=1)]
    layers += [nn.Conv2d(512, num_classes * 6, 3, padding=1)]
    layers += [nn.Conv2d(256, num_classes * 6, 3, padding=1)]
    layers += [nn.Conv2d(256, num_classes * 4, 3, padding=1)]
    layers += [nn.Conv2d(256, num_classes * 4, 3, padding=1)]
    return nn.ModuleList(layers)


def locs():
    layers = []
    layers += [nn.Conv2d(512, 4 * 4, 3, padding=1)]
    layers += [nn.Conv2d(1024, 4 * 6, 3, padding=1)]
    layers += [nn.Conv2d(512, 4 * 6, 3, padding=1)]
    layers += [nn.Conv2d(256, 4 * 6, 3, padding=1)]
    layers += [nn.Conv2d(256, 4 * 4, 3, padding=1)]
    layers += [nn.Conv2d(256, 4 * 4, 3, padding=1)]
    return nn.ModuleList(layers)


def extra():
    layers = []
    layers += [nn.Conv2d(1024, 256, 1)]
    layers += [nn.Conv2d(256, 512, 3, stride=2, padding=1)]
    layers += [nn.Conv2d(512, 128, 1)]
    layers += [nn.Conv2d(128, 256, 3, stride=2, padding=1)]
    layers += [nn.Conv2d(256, 128, 1)]
    layers += [nn.Conv2d(128, 256, 3)]
    layers += [nn.Conv2d(256, 128, 1)]
    layers += [nn.Conv2d(128, 256, 3)]
    return nn.ModuleList(layers)
