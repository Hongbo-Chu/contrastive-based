import imp
from turtle import forward
import torch.nn as nn
import torch
from torch.nn.functional import normalize
from models_vit import vit_large_patch16
def ConvBlocks(input, output, kernel_size=2, stride=1):
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size, stride),
        nn.BatchNorm2d(output)
    )
# class MSResNet(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MSResNet, self).__init__()
#         self.conv1 = ConvBlocks()
#         self.conv2 = ConvBlocks
#         self.conv3 = ConvBlocks
#         self.conv4 = ConvBlocks
#         self.conv5 = ConvBlocks
#         self.linear = nn.Linear()
#         self.BN = nn.BatchNorm2d()
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         x4 = self.conv4(x3)
#         x5 = self.conv5(x4)
#         out = self.linear(x2)

class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.vit = vit_large_patch16
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.vit(x_i)
        h_j = self.vit(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
    
    def forward_instance(self, x):
        h = self.resnet(x)
        c = self.instance_projector(h)
        return c
