import torch
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
from datetime import datetime

from model.ResNet_VGG_models import *
from data import *
from utils import clip_gradient,adjust_lr
import os
from scipy import misc
import smoothness
from eval_func import *
from model.loss_helper import TreeEnergyLoss
import torch.optim.lr_scheduler as lr_scheduler



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--tree_weight', type=float, default=10, help='tree energy loss weight')
parser.add_argument('--min_mi_weight', type=float, default=0.005, help='min mi weight')
parser.add_argument('--batchsize', type=int, default=3, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
parser.add_argument('--edge_loss_weight', type=float, default=1.0, help='weight for edge loss')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
# build models
model = Generator2(channel=32)

model.cuda()
optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr, betas=[0.5, 0.999])
scheduler = lr_scheduler.StepLR(optimizer,step_size=1  ,gamma = 0.95) 


data_root = './train_data/' #'/home2/dataset/liaixuan/train/'
image_root = data_root + 'img/'
gt_root =  data_root + 'gt/'
mask_root =  data_root + 'mask/'
grayimg_root =  data_root + 'gray/'
depth_root =  data_root + 'depth/'
train_loader = get_loader_rgbd(image_root, gt_root, mask_root, grayimg_root,depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
print('total_step',total_step)

model_save_path = './models/'
print(' model_save_path = ', model_save_path)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

temp_save_path = model_save_path + '/temp/'
if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

CE = torch.nn.BCELoss()
smooth_loss = smoothness.smoothness_loss(size_average=True)
tree_loss_struc = TreeEnergyLoss()

def train(train_loader, model, optimizer, epoch):
    model.train()
    
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, masks, grays, depths = pack
        images = Variable(images)
        gts = Variable(gts)
        masks = Variable(masks)
        grays = Variable(grays)
        depths = Variable(depths)
        images = images.cuda()
        gts = gts.cuda()
        masks = masks.cuda()
        grays = grays.cuda()
        depths = depths.cuda()

        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / torch.sum(masks)

        ### pred
        sal_rgbd, sal_rgbd_ref, embed_feat_d_init, embed_feat_d_ref, sal_rgb, sal_rgb_ref, embed_feat_rgb_init, embed_feat_rgb_ref, mi_min_loss = model(
            images, depths)
        sal_rgb_pred = torch.sigmoid(sal_rgb)
        sal_rgb_ref_pred = torch.sigmoid(sal_rgb_ref)
        images_tree = F.interpolate(images, size=(embed_feat_rgb_init.shape[2], embed_feat_rgb_init.shape[3]),
                                    mode='bilinear', align_corners=False)

        # rgb tree_energy_loss
        unlabeled_RoIs = (masks == 0)
        unlabeled_RoIs = unlabeled_RoIs.squeeze(1)

        ## initial prediction
        sal_rgb_prob_tree = F.interpolate(sal_rgb_pred,
                                        size=(embed_feat_rgb_init.shape[2], embed_feat_rgb_init.shape[3]),
                                        mode='bilinear', align_corners=False)
        tree_loss_rgb = tree_loss_struc(sal_rgb_prob_tree, images_tree, embed_feat_rgb_init, unlabeled_RoIs,
                                        tree_weight=opt.tree_weight)
        #  rgb pce loss
        sal_rgb_prob = sal_rgb_pred * masks
        sal_loss_rgb = ratio * CE(sal_rgb_prob, gts * masks)
        bce_rgb_init = sal_loss_rgb + tree_loss_rgb

        ## refined prediction
        sal_rgb_ref_prob_tree = F.interpolate(sal_rgb_ref_pred,
                                            size=(embed_feat_rgb_init.shape[2], embed_feat_rgb_init.shape[3]),
                                            mode='bilinear', align_corners=False)
        tree_loss_rgb_ref = tree_loss_struc(sal_rgb_ref_prob_tree, images_tree, embed_feat_rgb_ref,
                                            unlabeled_RoIs,
                                            tree_weight=opt.tree_weight)
        #  rgb pce loss
        sal_rgb_ref_prob = sal_rgb_ref_pred * masks
        sal_loss_rgb_ref = ratio * CE(sal_rgb_ref_prob, gts * masks)
        bce_rgb_ref = sal_loss_rgb_ref + tree_loss_rgb_ref

        bce_rgb = 0.5 * (bce_rgb_init + bce_rgb_ref)

        sal_rgbd_pred = torch.sigmoid(sal_rgbd)
        sal_rgbd_ref_pred = torch.sigmoid(sal_rgbd_ref)
        images_tree = F.interpolate(images, size=(embed_feat_d_init.shape[2], embed_feat_d_init.shape[3]),
                                    mode='bilinear', align_corners=False)

        # rgbd tree_energy_loss
        unlabeled_RoIs = (masks == 0)
        unlabeled_RoIs = unlabeled_RoIs.squeeze(1)

        ## initial prediction
        sal_rgbd_prob_tree = F.interpolate(sal_rgbd_pred,
                                        size=(embed_feat_d_init.shape[2], embed_feat_d_init.shape[3]),
                                        mode='bilinear', align_corners=False)
        tree_loss_rgbd = tree_loss_struc(sal_rgbd_prob_tree, images_tree, embed_feat_d_init, unlabeled_RoIs,
                                        tree_weight=opt.tree_weight)
        # rgbd pce loss
        sal_rgbd_prob = sal_rgbd_pred * masks
        sal_loss_rgbd = ratio * CE(sal_rgbd_prob, gts * masks)
        bce_rgbd_init = sal_loss_rgbd + tree_loss_rgbd

        ## refined prediction
        sal_rgbd_ref_prob_tree = F.interpolate(sal_rgbd_ref_pred,
                                            size=(embed_feat_d_init.shape[2], embed_feat_d_init.shape[3]),
                                            mode='bilinear', align_corners=False)
        tree_loss_rgbd_ref = tree_loss_struc(sal_rgbd_ref_prob_tree, images_tree, embed_feat_d_ref,
                                            unlabeled_RoIs,
                                            tree_weight=opt.tree_weight)
        # rgbd pce loss
        sal_rgbd_ref_prob = sal_rgbd_ref_pred * masks
        sal_loss_rgbd_ref = ratio * CE(sal_rgbd_ref_prob, gts * masks)
        bce_rgbd_ref = sal_loss_rgbd_ref + tree_loss_rgbd_ref

        bce_rgbd = 0.5 * (bce_rgbd_init + bce_rgbd_ref)

        latent_loss = opt.min_mi_weight*mi_min_loss

        loss = bce_rgbd + bce_rgb + latent_loss
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], sal1_loss: {:0.4f}, tree_loss: {:0.4f}, sal2_loss: {:0.4f}, tree_loss2: {:0.4f}, min_loss: {:0.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step, sal_loss_rgb.data, tree_loss_rgb.data, sal_loss_rgbd.data, tree_loss_rgbd.data, latent_loss.data))


    if epoch % 10 == 0:
        torch.save(model.state_dict(), model_save_path + 'scribble_' + '%d'  % epoch  + '.pth')
    scheduler.step()


print("Scribble it!")
for epoch in range(1, opt.epoch+1):
    # adjust_lr(optimizer, lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)



