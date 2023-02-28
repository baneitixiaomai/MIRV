import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Pred
from data import get_loader_rgbd,test_dataset_rgbd
from utils import adjust_lr, AvgMeter
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation
from tools import *
from eval_func import *


model_save_path = './models/'
print(' model_save_path = ', model_save_path)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

temp_save_path = model_save_path + '/temp' +'/'
if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate')
parser.add_argument('--vae_loss_weight', type=float, default=5, help='vae loss weight')
parser.add_argument('--feat_channel', type=int, default=32, help='network feature channel')
parser.add_argument('--latent_dim', type=int, default=32, help='vae latent dim')
parser.add_argument('--batchsize', type=int, default=3, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=8, help='every n epochs decay learning rate')
parser.add_argument('--modal_loss', type=float, default=0.5, help='weight of the fusion modal')
parser.add_argument('--focal_lamda', type=int, default=1, help='lamda of focal loss')
parser.add_argument('--bnn_steps', type=int, default=6, help='BNN sampling iterations')
parser.add_argument('--lvm_steps', type=int, default=6, help='LVM sampling iterations')
parser.add_argument('--pred_steps', type=int, default=6, help='Predictive sampling iterations')
parser.add_argument('--smooth_loss_weight', type=float, default=0.4, help='weight of the smooth loss')
parser.add_argument('--reg_weight', type=float, default=1e-4, help='weight for regularization term')
parser.add_argument('--lat_weight', type=float, default=5.0, help='weight for latent loss')

opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = Pred(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)
scheduler = lr_scheduler.StepLR(generator_optimizer,step_size=1 ,gamma = 0.95)  #2000

data_root =  '../train_data/'
image_root = data_root + 'img/'
gt_root =  data_root + 'gt/'
mask_root =  data_root + 'mask/'
depth_root =  data_root + 'depth/'
pseudo_gt_root =  "../first_stage/models/results_train/"

train_loader = get_loader_rgbd(image_root, gt_root, mask_root, pseudo_gt_root,depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)


total_step = len(train_loader)

CE = torch.nn.BCELoss()
size_rates = [0.75,1,1.25]  # multi-scale training


def structure_loss(pred, mask, weight=None):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts
    if weight == None:
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    else:
        weit = 1 + 5 * weight

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

## linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


print("Let's go!")
for epoch in range(1, (opt.epoch+1)):
    # scheduler.step()
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts, masks, psus, depths = pack
            images = Variable(images)
            gts = Variable(gts)
            masks = Variable(masks)
            psus = Variable(psus)
            depths = Variable(depths)
            images = images.cuda()
            gts = gts.cuda()
            masks = masks.cuda()
            psus = psus.cuda()
            depths = depths.cuda()

            img_size = images.size(2) * images.size(3) * images.size(0)
            ratio = img_size / torch.sum(masks)


            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                masks = F.upsample(masks, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                depths = F.upsample(depths, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                psus = F.upsample(psus, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            pred_priorxd, pred_posteriorxd, latent_loss  = generator(images, depths, psus, training=True)

            ## scribble loss
            scribble_prior = ratio*CE(torch.sigmoid(pred_priorxd)*masks, gts*masks)
            scribble_post = ratio*CE(torch.sigmoid(pred_posteriorxd)*masks, gts*masks)


            ## l2 regularizer the inference model
            reg_loss = l2_regularisation(generator.x_encoder) + \
                    l2_regularisation(generator.d_encoder) + l2_regularisation(generator.y_encoder) + \
                    l2_regularisation(generator.xy_encoder) + l2_regularisation(generator.dy_encoder)+ \
                    l2_regularisation(generator.dec_pred_prior)+ l2_regularisation( generator.dec_pred_post)

            reg_loss = opt.reg_weight * reg_loss
            anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
            latent_loss = opt.lat_weight * anneal_reg * latent_loss

            ## post loss
            loss_post =  structure_loss(pred_posteriorxd, psus) 
            loss_prior = structure_loss(pred_priorxd, psus) 
            gen_loss_cvae =  latent_loss + loss_post + 3* scribble_post
            gen_loss_gsnn = opt.vae_loss_weight * loss_prior + 3* scribble_prior

            gen_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss 
            gen_loss.backward()
            generator_optimizer.step()

            if rate == 1:
                loss_record.update(gen_loss.data, opt.batchsize)


        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen_loss_cvae: {:0.4f}, gen_loss_gsnn: {:0.4f}, reg_loss: {:0.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step, gen_loss_cvae.data, gen_loss_gsnn.data, reg_loss.data))

    scheduler.step()

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), model_save_path + 'Model' + '_%d' % epoch + '_gen.pth')


