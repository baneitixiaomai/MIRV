import torch
import torch.nn as nn
import torchvision.models as models
from model.ResNet import B2_ResNet
from model.vgg import B2_VGG
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Parameter, Softmax
import torch.nn.functional as F
from model.HolisticAttention import HA
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
import numpy as np
from model.spectral_normalization import SpectralNorm

class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size, feat_size):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.channel = channels

        self.fc1_rgb1 = nn.Linear(channels * 1 * feat_size * feat_size, latent_size*10)
        self.fc1_depth1 = nn.Linear(channels * 1 * feat_size * feat_size, latent_size*10)

        self.fc1_mu1 = SpectralNorm(nn.Linear(latent_size*10, latent_size*2))
        self.fc1_mu2 = SpectralNorm(nn.Linear(latent_size*2, latent_size*10))
        self.fc1_logvar1 = SpectralNorm(nn.Linear(latent_size*10, latent_size * 2))
        self.fc1_logvar2 = SpectralNorm(nn.Linear(latent_size * 2, latent_size*10))

        self.fc1_mu1_depth = SpectralNorm(nn.Linear(latent_size*10, latent_size * 2))
        self.fc1_mu2_depth = SpectralNorm(nn.Linear(latent_size * 2, latent_size*10))
        self.fc1_logvar1_depth = SpectralNorm(nn.Linear(latent_size*10, latent_size * 2))
        self.fc1_logvar2_depth = SpectralNorm(nn.Linear(latent_size * 2, latent_size*10))

        self.feat_size = feat_size



        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = rgb_feat.view(-1, self.channel * 1 * self.feat_size * self.feat_size)
        depth_feat = depth_feat.view(-1, self.channel * 1 * self.feat_size * self.feat_size)


        ## rgb_feat and depth feat in vector
        rgb_feat = self.fc1_rgb1(rgb_feat)  ## B*latent_size
        depth_feat = self.fc1_depth1(depth_feat) ## B*latent_size
        rgb_feat = F.normalize(rgb_feat, p=2, dim=1)
        depth_feat = F.normalize(depth_feat, p=2, dim=1)

        ## RGB to Depth

        mu_rgb = self.fc1_mu2(self.relu(self.fc1_mu1(rgb_feat)))
        logvar_rgb = self.tanh(self.fc1_logvar2(self.relu(self.fc1_logvar1(rgb_feat))))

        sample_size = rgb_feat.shape[0]
        random_index = torch.randperm(sample_size).long()

        positive = - (mu_rgb - depth_feat) ** 2 / logvar_rgb.exp()
        negative = - (mu_rgb - depth_feat[random_index,:]) ** 2 / logvar_rgb.exp()
        upper_bound_depth_rgb = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound_depth_rgb/2


class GlobalStatisticsNetwork(nn.Module):
    """Global statistics network

    Args:
        feature_map_size (int): Size of input feature maps
        feature_map_channels (int): Number of channels in the input feature maps
        latent_dim (int): Dimension of the representationss
    """

    def __init__(
        self, feature_map_size: int, feature_map_channels: int, latent_dim: int
    ):

        super().__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(
            in_features=(feature_map_size ** 2 * feature_map_channels) + latent_dim,
            out_features=512,
        )
        self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()

    def forward(
        self, feature_map: torch.Tensor, representation: torch.Tensor
    ) -> torch.Tensor:
        feature_map = self.flatten(feature_map)
        x = torch.cat([feature_map, representation], dim=1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        global_statistics = self.dense3(x)

        return global_statistics


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

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

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

    def __init__(self, features, interp=True):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        self.interp_flag = interp

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        if (self.interp_flag):
            output = nn.functional.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=True
            )

        return output

class FeatureFusionBlock1(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock1, self).__init__()

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


class VGG_encoder(nn.Module):
    # resnet based encoder for rgb
    def __init__(self):
        super(VGG_encoder, self).__init__()
        self.feat_enc = B2_VGG()
        self.relu = nn.ReLU(inplace=True)

        # if self.training:
        #     self.initialize_weights()

    def forward(self, x):
        x1 = self.feat_enc.conv1(x)  ## 352*352*64
        x2 = self.feat_enc.conv2(x1)  ## 176*176*128
        x3 = self.feat_enc.conv3(x2)  ## 88*88*256
        x4 = self.feat_enc.conv4_1(x3)  ## 44*44*512
        x5 = self.feat_enc.conv5_1(x4)  ## 22*22*512
        
        x_fea = [x1,x2,x3,x4,x5]

        return x_fea, self.feat_enc


class Res50_encoder(nn.Module):
    # resnet based encoder for depth
    def __init__(self):
        super(Res50_encoder, self).__init__()
        self.feat_enc = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        x = self.feat_enc.conv1(x)
        x = self.feat_enc.bn1(x)
        x = self.feat_enc.relu(x)
        x = self.feat_enc.maxpool(x)
        x1 = self.feat_enc.layer1(x)  # 256 x 64 x 64
        x2 = self.feat_enc.layer2(x1)  # 512 x 32 x 32
        x3 = self.feat_enc.layer3_1(x2)  # 1024, 22, 22]
        x4 = self.feat_enc.layer4_1(x3)  # 2048 x 11, 11

        x_fea = [x1,x2,x3,x4]
        return x_fea, self.feat_enc

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.feat_enc.state_dict().items():
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
        assert len(all_params.keys()) == len(self.feat_enc.state_dict().keys())
        self.feat_enc.load_state_dict(all_params)



class Res50_Decoder(nn.Module):
    # resnet based decoder
    def __init__(self, channel = 32):
        super(Res50_Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

        self.conv4 = nn.Conv2d(2048, channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(1024, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, channel, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(256, channel, kernel_size=3, stride=1, padding=1)

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


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x_fea):
        x1,x2,x3,x4 = x_fea
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)  # [15, 32, 22, 22])

        conv4_feat = self.path4(conv4_feat)
        conv43 = self.path3(conv4_feat, conv3_feat)
        conv432 = self.path2(conv43, conv2_feat)  # [15, 32, 88, 88]
        conv4321 = self.path1(conv432, conv1_feat)  # [15, 32, 176, 176])
        # import pdb;pdb.setq_trace()

        pred = self.output_conv(conv4321)

        return pred, conv432, conv4_feat


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Fusion(nn.Module):
    # resnet based decoder
    def __init__(self):
        super(Fusion, self).__init__()

        self.conv_depth1 = BasicConv2d(512, 256, kernel_size=3, padding=1)
        self.conv_depth2 = BasicConv2d(1024, 512, kernel_size=3, padding=1)
        self.conv_depth3 = BasicConv2d(2048, 1024, kernel_size=3, padding=1)
        self.conv_depth4 = BasicConv2d(4096, 2048, kernel_size=3, padding=1)

    def forward(self, x_fea):
        x1,x2,x3,x4 = x_fea

        conv1_feat = self.conv_depth1(x1)
        conv2_feat = self.conv_depth2(x2)
        conv3_feat = self.conv_depth3(x3)
        conv4_feat = self.conv_depth4(x4)  
        pred = [conv1_feat,conv2_feat,conv3_feat,conv4_feat]
        return pred

class Fusion_mix(nn.Module):
    # resnet based decoder
    def __init__(self):
        super(Fusion_mix, self).__init__()

        self.conv_depth1 = BasicConv2d(64+256, 64, kernel_size=3, padding=1)
        self.conv_depth2 = BasicConv2d(128+512, 128, kernel_size=3, padding=1)
        self.conv_depth3 = BasicConv2d(256+1024, 256, kernel_size=3, padding=1)
        self.conv_depth4 = BasicConv2d(512+2048, 512, kernel_size=3, padding=1)
        self.conv_depth5 = BasicConv2d(512+2048, 512, kernel_size=3, padding=1)

    def forward(self, x_fea):
        x1,x2,x3,x4,x5 = x_fea

        conv1_feat = self.conv_depth1(x1)
        conv2_feat = self.conv_depth2(x2)
        conv3_feat = self.conv_depth3(x3)
        conv4_feat = self.conv_depth4(x4)
        conv5_feat = self.conv_depth5(x5)
        pred = [conv1_feat,conv2_feat,conv3_feat,conv4_feat,conv5_feat]
        return pred

class Interp_to_specific_size(nn.Module):
    def __init__(self, interp_size):
        super(Interp_to_specific_size, self).__init__()
        self.interp_size = interp_size

    def forward(self, x):
        x = F.upsample(x, size=(self.interp_size, self.interp_size), mode='bilinear', align_corners=True)
        return x

def concate(x1,x2):
    mm = []
    if len(x1)==len(x2):
        temp = Interp_to_specific_size(x2[0].shape[2])(x1[0])
        ccc = torch.cat((temp,x2[0]),1)
        mm.append(ccc)
        for kkk in range(1,len(x1)):
            temp = Interp_to_specific_size(x2[kkk].shape[2])(x1[kkk])
            ccc = torch.cat((temp,x2[kkk]),1)
            mm.append(ccc)
        return mm


class DJSLoss(nn.Module):
    """Jensen Shannon Divergence loss"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, T: torch.Tensor, T_prime: torch.Tensor) -> float:
        """Estimator of the Jensen Shannon Divergence see paper equation (2)

        Args:
            T (torch.Tensor): Statistique network estimation from the marginal distribution P(x)P(z)
            T_prime (torch.Tensor): Statistique network estimation from the joint distribution P(xz)

        Returns:
            float: DJS estimation value
        """
        joint_expectation = (-F.softplus(-T)).mean()
        marginal_expectation = F.softplus(T_prime).mean()
        mutual_info = joint_expectation - marginal_expectation

        return -mutual_info


class Generator2(nn.Module):
    # resnet based decoder
    def __init__(self, channel):
        super(Generator2, self).__init__()
        self.rgb_encoder = Res50_encoder()
        self.depth_encoder = VGG_encoder()
        self.rgb_decoder = Res50_Decoder(channel)
        self.depth_decoder = VGG_Decoder(channel)
        self.fusion = Fusion_mix()

        self.latent_dim = 10
        self.image_size = 40

        self.rgbd_fusion =nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
        )
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

        self.harefine1 = HA_Refine_Resnet()
        self.harefine2 = HA_Refine_VGG()

        self.mutual_mi_min = Mutual_info_reg(channel, channel, self.latent_dim, self.image_size)

        self.resize = Interp_to_specific_size(self.image_size)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, depth):

          
        # rgb branch
        x_rgb, rgb_backbone = self.rgb_encoder(x)
        x1, x2, x3, x4 = x_rgb
        rgb_pred, rgb_fea_init,_ = self.rgb_decoder(x_rgb)
        x2_2, x3_2, x4_2 = self.harefine1(rgb_backbone, x2, rgb_pred)
        x_rgb_ref = [x1, x2_2, x3_2, x4_2]
        rgb_ref_pred, rgb_fea_ref, rgb_feat_ref = self.rgb_decoder(x_rgb_ref)

        # rgbd branch
        rgbd_img = depth 
        x_rgbd_init, depth_backbone = self.depth_encoder(rgbd_img)

        x_nr = [x1, x2_2, x3_2, x4_2, x4_2]

        x_rgbd = concate(x_nr,x_rgbd_init)
        x_rgbd = self.fusion(x_rgbd)
        x1_rgbd, x2_rgbd, x3_rgbd, x4_rgbd, x5_rgbd = x_rgbd

        rgbd_pred, rgbd_fea_init, _ = self.depth_decoder(x1_rgbd, x2_rgbd, x3_rgbd, x4_rgbd, x5_rgbd)
        x3_2_rgbd, x4_2_rgbd, x5_2_rgbd = self.harefine2(depth_backbone, x3_rgbd, rgbd_pred)
        rgbd_ref_pred, rgbd_fea_ref, rgbd_feat_ref = self.depth_decoder(x1_rgbd, x2_rgbd, x3_2_rgbd, x4_2_rgbd, x5_2_rgbd)

        ## cross-modal mutual information minimization rgb with rgbd
        mi_min_loss =0.5*( self.mutual_mi_min(self.resize(rgb_fea_ref), self.resize(rgbd_fea_ref)) + \
            self.mutual_mi_min(self.resize(rgb_fea_init), self.resize(rgbd_fea_init))) 

        return rgbd_pred, rgbd_ref_pred, rgbd_fea_init, rgbd_fea_ref, rgb_pred, rgb_ref_pred, rgb_fea_init, rgb_fea_ref, mi_min_loss


class VGG_Decoder(nn.Module):
    def __init__(self, channel=32):
        super(VGG_Decoder, self).__init__()
        self.conv5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 512)
        self.conv4 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 512)
        self.conv3 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 256)
        self.conv2 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 128)
        self.conv1 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 64)

        self.path5 = FeatureFusionBlock(channel)
        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel,interp=False)

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x1,x2,x3,x4,x5):
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)
        conv5_feat = self.conv5(x5)
        # conv1_feat = x1
        # conv2_feat = x2
        # conv3_feat = x3
        # conv4_feat = x4
        # conv5_feat = x5

        conv5_feat = self.path5(conv5_feat)
        conv54 = self.path4(conv5_feat,conv4_feat)
        conv543 = self.path3(conv54, conv3_feat)
        conv5432 = self.path2(conv543, conv2_feat)
        conv54321 = self.path1(conv5432, conv1_feat)

        pred = self.output_conv(conv54321)

        return pred, conv5432, conv5_feat

class HA_Refine_Resnet(nn.Module):
    def __init__(self):
        super(HA_Refine_Resnet, self).__init__()
        self.HA = HA()
        self.upsample0125 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)

    def forward(self, backbone, x2, initial_sal):
        x2_2 = self.HA(self.upsample0125(initial_sal).sigmoid(), x2)
        x3_2 = backbone.layer3_2(x2_2)  # 1024 x 16 x 16
        x4_2 = backbone.layer4_2(x3_2)  # 2048 x 8 x 8
        return x2_2, x3_2, x4_2


class HA_Refine_VGG(nn.Module):
    def __init__(self):
        super(HA_Refine_VGG, self).__init__()
        self.HA = HA()
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

    def forward(self, backbone, x3, initial_sal):
        x3_2 = self.HA(self.upsample025(initial_sal).sigmoid(), x3)
        x4_2 = backbone.conv4_2(x3_2)  # 1024 x 16 x 16
        x5_2 = backbone.conv5_2(x4_2)  # 2048 x 8 x 8
        return x3_2, x4_2, x5_2



