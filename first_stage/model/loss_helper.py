##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Copyright (c) 2022 megvii-model. All Rights Reserved.
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D

class TreeEnergyLoss(nn.Module):
    def __init__(self):
        super(TreeEnergyLoss, self).__init__()

        # self.weight =4 # self.configer.get('tree_loss', 'params')['weight']    0.4 
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma= 0.002)  # 0.002

    def forward(self, preds, low_feats, high_feats, ROIs,tree_weight):
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, _, h, w = preds.size()
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            ROIs = F.interpolate(ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
            N = ROIs.sum()

        prob = preds
        # prob = torch.softmax(preds, dim=1)


        # low-level MST
        tree = self.mst_layers(low_feats)  #torch.Size([13, 123903, 2])
        
        AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

        # high-level MST
        if high_feats is not None:
            tree = self.mst_layers(high_feats)
            AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False)  # [b, n, h, w]
            # AS: torch.Tensor = AS.to(low_feats.device)
        tree_loss = (ROIs * torch.abs(prob - AS)).sum()
        if N > 0:
            tree_loss /= N

        return tree_weight * tree_loss


class TreeEnergyLoss2(nn.Module):
    def __init__(self):
        super(TreeEnergyLoss2, self).__init__()

        # self.weight =4 # self.configer.get('tree_loss', 'params')['weight']    0.4 
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma= 0.002)  # 0.002

    def forward(self, preds, low_feats, high_feats, ROIs,tree_weight, pred_pred):
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, _, h, w = preds.size()
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            ROIs = F.interpolate(ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
            N = ROIs.sum()

        prob = preds
        # prob = torch.softmax(preds, dim=1)


        # low-level MST
        tree = self.mst_layers(low_feats)  #torch.Size([13, 123903, 2])
        
        AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

        # high-level MST
        if high_feats is not None:
            tree = self.mst_layers(high_feats)
            AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False)  # [b, n, h, w]
            # AS: torch.Tensor = AS.to(low_feats.device)
        tree_loss = (ROIs * torch.abs(pred_pred - AS)).sum()
        if N > 0:
            tree_loss /= N

        return tree_weight * tree_loss
