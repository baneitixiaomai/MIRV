import torch
import torch.nn as nn
import torchvision.models as models
from model.ResNet import B2_ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Parameter, Softmax
import torch.nn.functional as F
from model.HolisticAttention import HA
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
import numpy as np



class Encoder_x(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        # self.bn1_1 = nn.BatchNorm1d(latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        # self.bn2_1 = nn.BatchNorm1d(latent_size)

        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        # self.bn1_2 = nn.BatchNorm1d(latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        # self.bn2_2 = nn.BatchNorm1d(latent_size)

        self.fc1_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        # self.bn1_3 = nn.BatchNorm1d(latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        # self.bn2_3 = nn.BatchNorm1d(latent_size)

        self.tanh = torch.nn.Tanh()


        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))

        # print(output.size())
        if input.shape[2] == 256:
            # print('************************256********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 8 * 8)

            mu = self.fc1_1(output)
            logvar = self.fc2_1(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        elif input.shape[2] == 352:
            # print('************************352********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 11 * 11)

            mu = self.fc1_2(output)
            logvar = self.fc2_2(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        else:
            # print('************************bigger********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 14 * 14)

            mu = self.fc1_3(output)
            logvar = self.fc2_3(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar

class Encoder_xs(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_xs, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        # self.bn1_1 = nn.BatchNorm1d(latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        # self.bn2_1 = nn.BatchNorm1d(latent_size)

        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        # self.bn1_2 = nn.BatchNorm1d(latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        # self.bn2_2 = nn.BatchNorm1d(latent_size)

        self.fc1_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        # self.bn1_3 = nn.BatchNorm1d(latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        # self.bn2_3 = nn.BatchNorm1d(latent_size)

        self.tanh = torch.nn.Tanh()


        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))

        # print(output.size())
        if input.shape[2] == 256:
            # print('************************256********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 8 * 8)

            mu = self.fc1_1(output)
            logvar = self.fc2_1(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        elif input.shape[2] == 352:
            # print('************************352********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 11 * 11)

            mu = self.fc1_2(output)
            logvar = self.fc2_2(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        else:
            # print('************************bigger********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 14 * 14)

            mu = self.fc1_3(output)
            logvar = self.fc2_3(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar

class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(channels, 2*channels)
        self.leakyrelu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(2*channels)
        self.fc2 = nn.Linear(2*channels, channels)
        self.leakyrelu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(channels)
        self.fc3 = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)

        return x

class Encoder_xz(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_xz, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        # self.bn1_1 = nn.BatchNorm1d(latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        # self.bn2_1 = nn.BatchNorm1d(latent_size)

        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        # self.bn1_2 = nn.BatchNorm1d(latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        # self.bn2_2 = nn.BatchNorm1d(latent_size)

        self.fc1_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        # self.bn1_3 = nn.BatchNorm1d(latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        # self.bn2_3 = nn.BatchNorm1d(latent_size)

        self.tanh = torch.nn.Tanh()


        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))

        # print(output.size())
        if input.shape[2] == 256:
            # print('************************256********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 8 * 8)

            mu = self.fc1_1(output)
            logvar = self.fc2_1(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        elif input.shape[2] == 352:
            # print('************************352********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 11 * 11)

            mu = self.fc1_2(output)
            logvar = self.fc2_2(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        else:
            # print('************************bigger********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 14 * 14)

            mu = self.fc1_3(output)
            logvar = self.fc2_3(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar


class Encoder_xys(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_xys, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        # self.bn1_1 = nn.BatchNorm1d(latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        # self.bn2_1 = nn.BatchNorm1d(latent_size)

        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        # self.bn1_2 = nn.BatchNorm1d(latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        # self.bn2_2 = nn.BatchNorm1d(latent_size)

        self.fc1_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        # self.bn1_3 = nn.BatchNorm1d(latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        # self.bn2_3 = nn.BatchNorm1d(latent_size)

        self.tanh = torch.nn.Tanh()

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        output = self.leakyrelu(self.bn1(self.layer1(x)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))
        # print(output.size())
        if x.shape[2] == 256:
            # print('************************256********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 8 * 8)

            mu = self.fc1_1(output)
            logvar = self.fc2_1(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        elif x.shape[2] == 352:
            # print('************************352********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 11 * 11)

            mu = self.fc1_2(output)
            logvar = self.fc2_2(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        else:
            # print('************************bigger********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 14 * 14)

            mu = self.fc1_3(output)
            logvar = self.fc2_3(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar


class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x



class Image_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self):
        super(Image_encoder, self).__init__()
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)

        if self.training:
            self.initialize_weights()

    def forward(self, x, y=None, training=True):
        raw_x = x
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8
        return raw_x, x1, x2, x3, x4

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)


class Depth_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self):
        super(Depth_encoder, self).__init__()
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)

        if self.training:
            self.initialize_weights()

    def forward(self, x, y=None, training=True):
        raw_x = x
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8
        return raw_x, x1, x2, x3, x4

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        T = 1 / (var + eps)  # precision of i-th Gaussian expert at point x
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


def prior_expert(size):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))

    mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


class Pred(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel, latent_dim):
        super(Pred, self).__init__()
        # define q(z|x_i)
        self.image_encoder  = Image_encoder()
        self.depth_encoder  = Depth_encoder()

        self.x_encoder = Encoder_x(3, channel, latent_dim)
        self.d_encoder = Encoder_x(3, channel, latent_dim)
        self.y_encoder = Encoder_x(1, channel, latent_dim)
        
        self.xy_encoder = Encoder_xys(4, channel, latent_dim)
        self.dy_encoder = Encoder_xys(4, channel, latent_dim)

        # define p(x_i|z)
        self.dec_pred_prior = Pred_decoder_cat(channel, latent_dim)
        self.dec_pred_post = Pred_decoder_cat(channel, latent_dim)


        self.experts = ProductOfExperts()
        self.latent_dim = latent_dim

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x, dep, y=None, training=True):

        raw_d, d1, d2, d3, d4 = self.depth_encoder(dep)
        raw_x, x1, x2, x3, x4 = self.image_encoder(x)
        # prior P(z|X)
        mu, logvar = prior_expert((1, raw_x.shape[0], self.latent_dim)) # priorz [1, b, 32]
        priorx, mux, logvarx = self.x_encoder(raw_x) # priorx
        priord, mud, logvard = self.d_encoder(raw_d) # priord  [b, 32]

        muxd = torch.cat((mu, mux.unsqueeze(0), mud.unsqueeze(0)), dim=0)  # mux.unsqueeze(0) torch.Size([1, b, 32])
        logvarxd = torch.cat((logvar, logvarx.unsqueeze(0), logvard.unsqueeze(0)), dim=0)
        muxd, logvarxd = self.experts(muxd, logvarxd) # [b, 32]
        z_priorxd = self.reparametrize(muxd, logvarxd)
        priorxd = Independent(Normal(loc=muxd, scale=torch.exp(logvarxd)), 1)
        pred_priorxd = self.dec_pred_prior(torch.cat((x1,d1),dim=1),torch.cat((x2,d2),dim=1),torch.cat((x3,d3),dim=1),torch.cat((x4,d4),dim=1), z_priorxd)

        if training:
            # posterior q(z|X,y)
            posteriory, muy, logvary = self.y_encoder(y)  # posterior q(z|y)
            posteriorx, muxy, logvarxy = self.xy_encoder(torch.cat((raw_x, y), 1)) # posterior q(z|xr,y)
            posteriord, mudy, logvardy = self.dy_encoder(torch.cat((raw_d, y), 1)) # posterior q(z|xd,y)
            muxdy = torch.cat((muy.unsqueeze(0), muxy.unsqueeze(0), mudy.unsqueeze(0)), dim=0)
            logvarxdy = torch.cat((logvary.unsqueeze(0), logvarxy.unsqueeze(0), logvardy.unsqueeze(0)), dim=0)
            muxdy, logvarxdy = self.experts(muxdy, logvarxdy)
            z_posteriorxdy = self.reparametrize(muxdy, logvarxdy)
            posteriorxdy = Independent(Normal(loc=muxdy, scale=torch.exp(logvarxdy)), 1)

            # p(y|xr,z) , p(y|xd,z)
            pred_posteriorxd = self.dec_pred_post(torch.cat((x1,d1),dim=1),torch.cat((x2,d2),dim=1),torch.cat((x3,d3),dim=1),torch.cat((x4,d4),dim=1), z_posteriorxdy)

            # KL div
            latent_lossx = torch.mean(self.kl_divergence(posteriorx, priorx)) 
            latent_lossd = torch.mean(self.kl_divergence(posteriord, priord)) 
            latent_lossxdy = torch.mean(self.kl_divergence(posteriorxdy, priorxd)) 
            latent_loss = latent_lossx + latent_lossd + latent_lossxdy
            
            return pred_priorxd, pred_posteriorxd, latent_loss

        else:
            return pred_priorxd




class Pred_decoder_cat(nn.Module):
    def __init__(self, channel, latent_dim):
        super(Pred_decoder_cat, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048*2)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024*2)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512*2)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256*2)

        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel)

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        self.noise_conv1 = nn.Conv2d(channel + latent_dim, channel, kernel_size=3, padding=1)
        self.noise_conv2 = nn.Conv2d(channel + latent_dim, channel, kernel_size=3, padding=1)
        self.noise_conv3 = nn.Conv2d(channel + latent_dim, channel, kernel_size=3, padding=1)
        self.noise_conv4 = nn.Conv2d(channel + latent_dim, channel, kernel_size=3, padding=1)
        self.spatial_axes = [2, 3]


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            device)
        return torch.index_select(a, dim, order_index)

    def KL_computation(self, inf_mean, inf_logvar):
        KLD = -1 * 0.5 * torch.sum(1 + inf_logvar - inf_mean.pow(2) - inf_logvar.exp())
        return KLD

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu)

    def forward(self, x1,x2,x3,x4, z=None):
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)

        if z==None:
            conv4_feat = self.path4(conv4_feat)
            conv43 = self.path3(conv4_feat, conv3_feat)
            conv432 = self.path2(conv43, conv2_feat)
            conv4321 = self.path1(conv432, conv1_feat)
            pred = self.output_conv(conv4321)
            return pred
        else:
            z_noise = torch.unsqueeze(z, 2)
            z_noise = self.tile(z_noise, 2, conv4_feat.shape[self.spatial_axes[0]])
            z_noise = torch.unsqueeze(z_noise, 3)
            z_noise = self.tile(z_noise, 3, conv4_feat.shape[self.spatial_axes[1]])

            conv4_feat = torch.cat((conv4_feat, z_noise), 1)
            conv4_feat = self.noise_conv4(conv4_feat)

            z_noise = torch.unsqueeze(z, 2)
            z_noise = self.tile(z_noise, 2, conv3_feat.shape[self.spatial_axes[0]])
            z_noise = torch.unsqueeze(z_noise, 3)
            z_noise = self.tile(z_noise, 3, conv3_feat.shape[self.spatial_axes[1]])

            conv3_feat = torch.cat((conv3_feat, z_noise), 1)
            conv3_feat = self.noise_conv3(conv3_feat)

            z_noise = torch.unsqueeze(z, 2)
            z_noise = self.tile(z_noise, 2, conv2_feat.shape[self.spatial_axes[0]])
            z_noise = torch.unsqueeze(z_noise, 3)
            z_noise = self.tile(z_noise, 3, conv2_feat.shape[self.spatial_axes[1]])

            conv2_feat = torch.cat((conv2_feat, z_noise), 1)
            conv2_feat = self.noise_conv2(conv2_feat)

            z_noise = torch.unsqueeze(z, 2)
            z_noise = self.tile(z_noise, 2, conv1_feat.shape[self.spatial_axes[0]])
            z_noise = torch.unsqueeze(z_noise, 3)
            z_noise = self.tile(z_noise, 3, conv1_feat.shape[self.spatial_axes[1]])

            conv1_feat = torch.cat((conv1_feat, z_noise), 1)
            conv1_feat = self.noise_conv1(conv1_feat)


            conv4_feat = self.path4(conv4_feat)
            conv43 = self.path3(conv4_feat, conv3_feat)
            conv432 = self.path2(conv43, conv2_feat)
            conv4321 = self.path1(conv432, conv1_feat)

            pred = self.output_conv(conv4321)
            return pred
