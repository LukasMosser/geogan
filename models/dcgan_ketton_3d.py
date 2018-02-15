import torch.nn as nn


class DCGAN3D_G(nn.Module):
    def __init__(self):
        super(DCGAN3D_G, self).__init__()
        self.main = self.build_network()

    def build_network(self):
        blocks = []
        blocks += [nn.ConvTranspose3d(512, 512, kernel_size=4, stride=1, bias=False),
                   nn.BatchNorm3d(512), nn.LeakyReLU(negative_slope=0.2)]
        blocks += [nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                   nn.BatchNorm3d(256), nn.LeakyReLU(negative_slope=0.2)]
        blocks += [nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                   nn.BatchNorm3d(128), nn.LeakyReLU(negative_slope=0.2)]
        blocks += [nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                   nn.BatchNorm3d(64), nn.LeakyReLU(negative_slope=0.2)]
        blocks += [nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                   nn.BatchNorm3d(64), nn.LeakyReLU(negative_slope=0.2)]
        blocks += [nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
                   nn.Tanh()]
        return nn.Sequential(*blocks)

    def forward(self, z):
        x = self.main(z)
        return x


class DCGAN3D_D(nn.Module):
    def __init__(self):
        super(DCGAN3D_D, self).__init__()
        self.main = self.build_network()

    def build_network(self):
        blocks = []
        blocks += [nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
                   nn.LeakyReLU(negative_slope=0.2)]
        blocks += [nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                   nn.BatchNorm3d(128), nn.LeakyReLU(negative_slope=0.2)]
        blocks += [nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                   nn.BatchNorm3d(256), nn.LeakyReLU(negative_slope=0.2)]
        blocks += [nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                   nn.BatchNorm3d(512), nn.LeakyReLU(negative_slope=0.2)]
        blocks += [nn.Conv3d(512, 1, kernel_size=4, stride=1, bias=False),
                   nn.Sigmoid()]
        return nn.Sequential(*blocks)

    def forward(self, x):
        label = self.main(x)
        return label